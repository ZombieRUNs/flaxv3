from typing import List,Dict,Tuple,Callable
import math

from flax import nnx
import jax 
import jax.numpy as jnp
import numpy as np
from elements import Space
from einops import rearrange, reduce

from .net import DynGRU,MLP,MiniGru

sg = jax.lax.stop_gradient

def mask(xs:Dict|Tuple, mask:jax.Array):
    def fn(x, y):
        assert x.shape == y.shape
        expanded = jnp.expand_dims(mask,list(range(mask.ndim,x.ndim)))
        return jnp.where(expanded,x,y)
    ys = jax.tree.map(lambda x:jnp.zeros_like(x),xs)
    return jax.tree.map(fn,xs,ys)

def kl(logits1:jax.Array,logits2:jax.Array,free_bits:float=1.,unimix:float=0.01):
    logits1 = get_unimix_logits(logits1,unimix)
    logits2 = get_unimix_logits(logits2,unimix)
    logprob1 = jax.nn.log_softmax(logits1, -1)
    logprob2 = jax.nn.log_softmax(logits2, -1)
    prob = jax.nn.softmax(logits1)
    kl_loss = (prob*(logprob1 - logprob2)).sum(-1)
    kl_loss = reduce(kl_loss, 'B L D -> B L',"sum")
    kl_loss = jnp.mean(kl_loss)
    kl_loss = jnp.maximum(kl_loss, free_bits)
    return kl_loss

def get_unimix_logits(logits:jax.Array, unimix:float=0.01):
    probs = jax.nn.softmax(logits, axis=-1)
    uniform = jnp.ones_like(probs)/logits.shape[-1]
    probs = (1 - unimix)*probs + unimix*uniform
    return jnp.log(probs)

class Encoder(nnx.Module):
    def __init__(self,image_shape:tuple,depths:int,mults:tuple,act:str,norm:str,init_key:nnx.Rngs):
        super().__init__()
        conv_modules = []
        norm_modules = []
        final_shape = [*image_shape]
        depths_list = [image_shape[-1]]+[depths*mult for mult in mults]
        for in_features, out_features in zip(depths_list[:-1],depths_list[1:]):
            conv_modules.append(nnx.Conv(in_features,out_features,(4,4),(2,2),rngs=init_key))
            norm_modules.append(getattr(nnx,norm)(out_features,rngs=init_key))
            final_shape[-2] = final_shape[-2]//2
            final_shape[-3] = final_shape[-3]//2
        final_shape[-1] = depths*mults[-1]
        self.conv_modules = conv_modules
        self.norm_modules = norm_modules
        self.act = getattr(nnx,act)
        self.final_shape = final_shape
        
    def __call__(self, x:jax.Array):
        x = jax.device_put(x)
        x = x.astype(jnp.float32)/255. - 0.5
        for conv, norm in zip(self.conv_modules,self.norm_modules):
            x = self.act(norm(conv(x)))
        return x

class Decoder(nnx.Module):
    def __init__(self,feats_dim:int,final_shape:tuple,channels:int,depths:int,mults:tuple,act:str,norm:str,init_key:nnx.Rngs):
        super().__init__()
        final_shape = list(final_shape)
        self.final_shape = final_shape
        self.image_proj = nnx.Linear(feats_dim, math.prod(final_shape), rngs=init_key)
        self.init_norm = getattr(nnx, norm)(final_shape[-1], rngs=init_key)
        depths_list = [channels] + [depths*mult for mult in mults]
        depths_list = depths_list[::-1]
        dconv_modules,norm_modules = [], []
        for in_features, out_features in zip(depths_list[:-2],depths_list[1:-1]):
            dconv_modules.append(nnx.ConvTranspose(in_features,out_features,(4,4),(2,2),rngs=init_key))
            norm_modules.append(getattr(nnx,norm)(out_features,rngs=init_key))
        dconv_modules.append(nnx.ConvTranspose(depths_list[-2],depths_list[-1],(4,4),(2,2),rngs=init_key))
        self.dconv_modules = dconv_modules
        self.norm_modules = norm_modules
        self.act = getattr(nnx,act)
        
    def __call__(self, x:jax.Array):
        x = self.image_proj(x)
        x = rearrange(x, '... (H W C) -> ... H W C',H=self.final_shape[0],W=self.final_shape[1])
        x = self.act(self.init_norm(x))
        for dconv, norm in zip(self.dconv_modules[:-1], self.norm_modules):
            x = self.act(norm(dconv(x)))
        return jax.nn.sigmoid(self.dconv_modules[-1](x))
    
class RSSM(nnx.Module):
    def __init__(self,
                 final_shape:List,
                 hidden:int,
                 layers:int,
                 deter:int,
                 group:int,
                 stoch:int,
                 classes:int,
                 action_space:Space,
                 act:str,
                 norm:str,
                 init_key:nnx.Rngs
                 ):
        super().__init__()
        action_dim = action_space.high.item() if action_space.discrete else action_space.shape[-1]
        self.action_dim = action_dim
        self.action_space = action_space
        self.stoch_dim,self.classes,self.deter = stoch,classes,deter   
        # self._core = DynGRU(deter,stoch*classes,action_dim,hidden,layers,group,act,norm,init_key)
        self._core = MiniGru(deter,stoch*classes,action_dim,hidden,act,norm,init_key)
        self.imgmlp = MLP(deter,hidden,hidden,2,act,norm,init_key)
        self.obsmlp = MLP(deter+math.prod(final_shape),hidden,hidden,1,act,norm,init_key)
        self.prior_proj = nnx.Linear(hidden,stoch*classes,rngs=init_key)
        self.post_proj = nnx.Linear(hidden, stoch*classes, rngs=init_key)
    
    def init_recurrent(self,batch_size:int, key:jax.random.PRNGKey):
        carry = dict(stoch = jnp.zeros((batch_size,self.stoch_dim*self.classes)),
                    deter = jnp.zeros((batch_size,self.deter)),
                    key = key) 
        return carry
    
    def action_embed(self,action:jax.Array):
        if self.action_space.discrete:
            action = jax.nn.one_hot(action, self.action_space.high, dtype=jnp.float32)
        else:
            action = action.astype(jnp.float32)
        return action 

    def feat2carry(self, feats:Dict, key:jax.random.PRNGKey):
        carry = dict(deter=feats['deter'],stoch=feats['stoch'])
        carry = jax.tree.map(lambda x:rearrange(x,'B L ... -> (B L) ...'), carry)
        carry['key'] = key
        return sg(carry) 

    def straight_through_sample(self, logits:jax.Array, key:jax.random.PRNGKey):
        new_key, sub_key = jax.random.split(key,2)
        logits = get_unimix_logits(logits)
        sample = jax.random.categorical(sub_key,logits,axis=-1)
        sample = jax.nn.one_hot(sample, logits.shape[-1])
        probs = jax.nn.softmax(logits,axis=-1)
        return sample + probs - sg(probs), new_key
    
    def _post(self, tokens:jax.Array, deter:jax.Array, key:jax.random.PRNGKey):
        x = jnp.concatenate([tokens,deter],axis=-1)
        post_logits = self.post_proj(self.obsmlp(x))
        post_logits = rearrange(post_logits,'... (C K) -> ... C K',C=self.stoch_dim)
        post_sample, new_key = self.straight_through_sample(post_logits, key)
        post_sample = rearrange(post_sample,'... C K -> ... (C K)')
        return post_sample, post_logits, new_key
    
    def _prior(self,deter:jax.Array, key:jax.random.PRNGKey):
        prior_logits = self.prior_proj(self.imgmlp(deter))    
        prior_logits = rearrange(prior_logits, '... (C K) -> ... C K', C=self.stoch_dim)
        prior_sample, new_key = self.straight_through_sample(prior_logits, key)
        prior_sample = rearrange(prior_sample,'... C K -> ... (C K)',C=self.stoch_dim)
        return prior_sample, new_key
        
    def _observe(self, carry:Dict, tokens:jax.Array, action:jax.Array, reset:jax.Array):
        con = ~reset
        # (carry, action) = mask((carry, action), con)
        (carry['deter'],carry['stoch'],action) = mask((carry['deter'],carry['stoch'],action),con)
        action = self.action_embed(action)
        action = mask(action, con)            
        deter = self._core(stoch=carry['stoch'], deter=carry['deter'], action=action)
        stoch, post_logits, new_key = self._post(tokens,deter,carry['key'])
        carry = dict(stoch=stoch,deter=deter,key=new_key)
        feat = dict(stoch=stoch,deter=deter,post_logits=post_logits)
        return carry, feat
    
    # def observe(self, carry:Dict, tokens:jax.Array, action:jax.Array, reset:jax.Array):
    #     feats = dict(stoch=[],deter=[],post_logits=[])
    #     for i in range(tokens.shape[1]):
    #         carry,feat = self._observe(carry,tokens[:,i],action[:,i],reset[:,i])
    #         feats['deter'].append(feat['deter'])
    #         feats['stoch'].append(feat['stoch'])
    #         feats['post_logits'].append(feat['post_logits'])
    #     feats['deter'] = jnp.stack(feats['deter'], axis=1)
    #     feats['stoch'] = jnp.stack(feats['stoch'], axis=1)
    #     feats['post_logits'] = jnp.stack(feats['post_logits'], axis=1)
    #     return carry, feats
    
    def observe(self, carry:Dict, tokens:jax.Array, action:jax.Array, reset:jax.Array):
        (tokens, action, reset) = jax.tree.map(lambda x:rearrange(x,'B L ... -> L B ...'),(tokens, action, reset))
        carry,feats = jax.lax.scan(lambda carry,input:self._observe(carry, *input), carry,(tokens,action,reset))
        feats = jax.tree.map(lambda x:rearrange(x,'L B ... -> B L ...'),feats)
        return carry, feats

    def imagine_1_step(self, carry:Dict, policy):
        pol_feat = jnp.concatenate([carry['deter'],carry['stoch']], axis=-1)
        action, carry['key'] = policy(pol_feat, carry['key']) 
        embd_action = self.action_embed(action)
        deter = self._core(deter=carry['deter'],stoch=carry['stoch'],action=embd_action)
        stoch, new_key = self._prior(deter, carry['key'])
        carry = dict(stoch=stoch,deter=deter,key=new_key)
        return carry, action
    
    def imagine(self, carry:Dict, policy:Callable, imagine_length:int):
        feats = dict(stoch=[],deter=[],action=[])
        feats['deter'].append(carry['deter'])
        feats['stoch'].append(carry['stoch'])
        for _ in range(imagine_length):
            carry,action = self.imagine_1_step(carry, policy)    
            feats['deter'].append(carry['deter'])
            feats['stoch'].append(carry['stoch'])
            feats['action'].append(action)
        feats['deter'] = jnp.stack(feats['deter'], axis=1)
        feats['stoch'] = jnp.stack(feats['stoch'], axis=1)
        feats['action'] = jnp.stack(feats['action'], axis=1)
        return carry, feats
    
    def compute_loss(self, carry:Dict, tokens:jax.Array, action:jax.Array, reset:jax.Array): #这里的tokens实际上领先与action，reset一个时间步
        carry,feats = self.observe(carry, tokens, action, reset) #B L D
        post_logits = feats['post_logits'][:,1:]
        prior_logits = self.prior_proj(self.imgmlp(feats['deter']))[:,1:]    
        prior_logits = rearrange(prior_logits, '... (C K) -> ... C K', C=self.stoch_dim)
        dyn_loss = kl(sg(post_logits), prior_logits)
        rep_loss = kl(post_logits, sg(prior_logits))
        losses = {'dyn_loss':dyn_loss,'rep_loss':rep_loss}
        return carry, feats, losses

if __name__ == '__main__':
    # import numpy as np
    # action_space = Space(dtype=np.uint8,shape=(),high=18,low=0)
    # init_key = nnx.Rngs(0)
    # enc = Encoder((64,64,3),3,(1,2,3,4),'silu','BatchNorm',init_key)
    # rssm = RSSM(enc.final_shape,1024,1,2048,8,32,32,action_space,'silu','LayerNorm',init_key)
    # dec = Decoder((64,64,3),32,(1,2,3,4),'silu','BatchNorm',init_key)
    # carry = rssm.init_recurrent(16,jax.random.PRNGKey(0))
    
    # image = jax.random.randint(init_key.noise(),(16,64,64,64,3),minval=0,maxval=255)
    # action = jax.random.randint(init_key.noise(),(16,64),minval=0,maxval=17)
    # reset =  np.random.randn(16,64) < 0.2
    # reset = jax.device_put(reset)
    # tokens = enc(image)
    # tokens = rearrange(tokens,'... H W C -> ... (H W C)')
    # carry, feats, losses = rssm.compute_loss(carry, tokens, action, reset)
    # #kankanshape
    # losses_shape = jax.tree.map(lambda x:x.shape, losses)
    # init_carry = rssm.feat2carry(feats, carry['key'])
    # carry_shape = jax.tree.map(lambda x:x.shape, init_carry)
    # print(carry_shape)
    # print(init_carry['key'])
    
    pass
    