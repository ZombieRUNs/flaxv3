import jax
import jax.numpy as jnp
import jax.random as random
from flax import nnx 
from flax.nnx.nn import dtypes
import flax
import numpy as np
from einops import rearrange

class MLP(nnx.Module):
    def __init__(self,
                 in_features:int,
                 out_features:int,
                 hidden_features:int,
                 layers:int,
                 act:str,
                 norm:str,
                 init_key:nnx.Rngs):
        super().__init__()
        features = [in_features] + [hidden_features for _ in range(layers-1)] + [out_features]
        self.layers = [nnx.Linear(in_feature,out_feature,rngs=init_key) 
                       for in_feature, out_feature in zip(features[:-1],features[1:])]
        self.norm = [getattr(nnx,norm)(out_feature,rngs=init_key) for out_feature in features[1:]]
        self.act = getattr(nnx,act)
        
    def __call__(self,x:jax.Array):
        for layer, norm in zip(self.layers,self.norm):
            x = self.act(norm(layer(x)))
        return x

class MLPHead(nnx.Module):
    def __init__(self,in_features:int,
                 out_features:int,
                 hidden_features:int,
                 layers:int,
                 act:str,
                 norm:str,
                 init_key:nnx.Rngs,
                 head_nums:int=1):
        super().__init__()
        self.backbone = MLP(in_features,hidden_features,hidden_features,layers-1,act,norm,init_key)
        self.head = [nnx.Linear(hidden_features, out_features,rngs=init_key) for _ in range(head_nums)]
    def __call__(self, x:jax.Array):
        x = self.backbone(x)
        out = [head(x) for head in self.head]
        return out
    
class BlockLinear(nnx.Module):
    def __init__(self, 
                 in_features:int, 
                 out_features:int, 
                 blocks:int, 
                 init_key:nnx.Rngs, 
                 bias:bool=True):
        super().__init__()
        assert in_features % blocks == 0 and out_features % blocks == 0
        self.blocks = blocks
        self.weight = nnx.Param(random.normal(init_key.noise(), (blocks, in_features//blocks, out_features//blocks)))
        self.bias = nnx.Param(jnp.zeros(out_features)) if bias else 0.
    
    def __call__(self, inputs:jax.Array):
        inputs, weight, bias = dtypes.promote_dtype(
      (
        inputs,
        self.weight.value,
        self.bias.value if self.bias is not None else self.bias,
      ),
    )
        inputs = inputs.reshape(*inputs.shape[:-1], self.blocks, inputs.shape[-1]//self.blocks)

        x = jnp.einsum('...ki,kio->...ko', inputs, weight)
        x = x.reshape(*x.shape[:-2], -1)
        x += bias
        return x

class BlockMLP(nnx.Module):
    def __init__(self,
                 in_features:int,
                 out_features:int,
                 hidden_features:int,
                 blocks:int,
                 layers:int,
                 act:str,
                 norm:str,
                 init_key:nnx.Rngs):
        super().__init__()
        features = [in_features] + [hidden_features for _ in range(layers-1)] + [out_features]
        self.layers = [BlockLinear(in_feature,out_feature,blocks,init_key=init_key) 
                       for in_feature, out_feature in zip(features[:-1],features[1:])]
        self.norm = [getattr(nnx,norm)(out_feature,rngs=init_key) for out_feature in features[1:]]
        self.act = getattr(nnx,act)
        
    def __call__(self,x:jax.Array):
        for layer, norm in zip(self.layers,self.norm):
            x = self.act(norm(layer(x)))
        return x
    
        
class DynGRU(nnx.Module):
    def __init__(self,
                 deter:int,
                 stoch:int,
                 action_dim:int,
                 hidden:int,
                 layers:int,
                 group:int,
                 act:str,
                 norm:str,
                 init_key:nnx.Rngs,):
        super().__init__()
        self.g = group
        common_params = {'act':act,'norm':norm,'init_key':init_key}
        self.deter_proj = MLP(in_features=deter, out_features=hidden, hidden_features=hidden, layers=1, **common_params)
        self.stoch_proj = MLP(in_features=stoch, out_features=hidden, hidden_features=hidden, layers=1, **common_params)
        self.action_proj = MLP(in_features=action_dim, out_features=hidden, hidden_features=hidden, layers=1, **common_params)
        self.flat2group = lambda x:rearrange(x,'... (g h) -> ... g h', g=group)
        self.group2flat = lambda x:rearrange(x,'... g h -> ... (g h)', g=group)
        self.gru_in_proj = BlockMLP(in_features=3*group*hidden+deter,out_features=deter,hidden_features=deter,blocks=group,
                                    layers=layers,**common_params)
        self.gru_out_proj = BlockLinear(deter,3*deter,group,init_key=init_key)
    
    def __call__(self,deter:jax.Array, stoch:jax.Array, action:jax.Array):
        action /= jax.lax.stop_gradient(jnp.maximum(1,jnp.abs(action)))
        x0 = self.deter_proj(deter)
        x1 = self.stoch_proj(stoch)
        x2 = self.action_proj(action)
        x = jnp.concatenate([x0,x1,x2],axis=-1)[...,None,:].repeat(self.g,-2)
        x = self.group2flat(jnp.concatenate([self.flat2group(deter),x],axis=-1))
        x = self.gru_in_proj(x)
        x = self.gru_out_proj(x)
        gates = jnp.split(self.flat2group(x),3,axis=-1)
        reset, cand, update = [self.group2flat(x) for x in gates]
        reset = nnx.sigmoid(reset)
        cand = nnx.tanh(reset*cand)
        update = nnx.sigmoid(update-1)
        deter = update*cand + (1-update)*deter
        return deter

class MiniGru(nnx.Module):
    def __init__(self,
                 deter:int,
                 stoch:int,
                 action_dim:int,
                 hidden:int,
                 act:str,
                 norm:str,
                 init_key:nnx.Rngs,):
        super().__init__()
        common_params = {'act':act,'norm':norm,'init_key':init_key}
        self.stoch_action_proj = MLP(in_features=stoch+action_dim, out_features=hidden, hidden_features=hidden, layers=1, 
                                     **common_params)
        self._core = nnx.Linear(hidden+deter,3*deter, use_bias=False, rngs=init_key)
        self._core_norm = nnx.LayerNorm(3*deter, epsilon=1e-03, rngs=init_key)
    
    def __call__(self, deter:jax.Array, stoch:jax.Array, action:jax.Array):
        action /= jax.lax.stop_gradient(jnp.maximum(1,jnp.abs(action)))
        tokens = jnp.concatenate([stoch,action],axis=-1)
        tokens = self.stoch_action_proj(tokens)
        parts = self._core_norm(self._core(jnp.concatenate([tokens, deter],axis=-1)))
        reset, cand, update = jnp.split(parts,3,axis=-1)
        reset = nnx.sigmoid(reset)
        cand = nnx.tanh(reset*cand)
        update = nnx.sigmoid(update-1.)
        deter = update*cand + (1-update)*deter
        return deter
            

class Harmonizer(nnx.Module):
    def __init__(self,):
       self.harmony_scale = nnx.Param(jnp.zeros((1,)))
    def __call__(self, loss, regularize=True):
        scale = jnp.squeeze(self.harmony_scale,axis=0)
        if regularize:
            return loss/(jnp.exp(scale)) + jnp.log(jnp.exp(scale) + 1)
        else:
            return loss/(jnp.exp(scale))
    @property
    def get_scale(self):
        return jax.lax.stop_gradient(self.harmony_scale.value).squeeze(0)

if __name__ == '__main__':
    # import optax
    # from tqdm import tqdm
    
    # init_key = nnx.Rngs(1)
    # # blocklinear = BlockLinear(128,128,8,init_key)
    # gru_module = DynGRU(2048,1024,18,256,1,8,'silu','RMSNorm',init_key)
    # optimizer = nnx.Optimizer(gru_module,optax.adam(1e-3))
    # deter = random.normal(init_key.noise(),shape=(4,2048))
    # stoch = random.normal(init_key.noise(),shape=(4,1024))
    # action = random.normal(init_key.noise(),shape=(4,18))
    
    # optimizer = nnx.Optimizer(gru_module,optax.adam(1e-3))
    
    # def loss_fn(model:DynGRU,deter,stoch,action):
    #     # target = jax.lax.stop_gradient(deter)
    #     target = deter*0.5
    #     return jnp.mean(jnp.sum((model(deter,stoch,action)-target)**2,axis=-1))
    
    # @nnx.jit
    # def update(model:DynGRU,optimizer,deter,stoch,action):
    #     loss,grad = nnx.value_and_grad(loss_fn)(model,deter,stoch,action)
    #     optimizer.update(grad)
    #     grad_norm = optax.global_norm(grad)
    #     return loss,grad_norm
        
    # with tqdm(total=100000) as pbar:
    #     for i in range(100000):
    #         loss,grad_norm = update(gru_module,optimizer,deter,stoch,action)
    #         pbar.set_postfix(dict(loss=loss,grad_norm=grad_norm.item()))
    #         pbar.update(1)
    
    pass