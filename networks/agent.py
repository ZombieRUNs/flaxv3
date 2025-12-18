import pathlib
import sys
from copy import deepcopy
import os

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import jax
import jax.numpy as jnp
from flax import nnx
from elements import Space
import optax
import orbax.checkpoint as ocp

from .net import MLPHead
from utils.functional import SymLogTwoHot,bounded_normal,Categorical,Percentile
from utils.optim import make_opt, make_simple_opt

sg = jax.lax.stop_gradient

def calc_lambda_return(rewards:jax.Array, values:jax.Array, termination:jax.Array, gamma:float, lmbda:float):
    inv_termination = (termination*-1.) + 1.
    
    batch_size, batch_length = rewards.shape[:2]
    gamma_return = jnp.zeros((batch_size,batch_length+1))
    # gamma_return[:,-1] = values[:,-1]
    gamma_return = gamma_return.at[:,-1].set(values[:,-1])
    for t in reversed(range(batch_length)):
        # gamma_return[:,t] = rewards[:,t] + gamma * inv_termination[:, t] * (1-lmbda) * values[:,t] + \
        #                     gamma * inv_termination[:, t] * lmbda * gamma_return[:, t+1]
        updates = rewards[:,t] + gamma * inv_termination[:, t] * (1-lmbda) * values[:,t] + \
                    gamma * inv_termination[:, t] * lmbda * gamma_return[:, t+1]
        gamma_return = gamma_return.at[:,t].set(updates)
    return gamma_return[:, :-1]

class ActorCritic(nnx.Module):
    def __init__(self,feat_dim:int, hidden_dim:int, layers:int, action_space:Space, 
                 gamma:float, lmbda:float, entropy_coef:float, act:str, norm:str,
                 init_key:nnx.Rngs):
        super().__init__()
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef
        self.discrete = action_space.discrete
        self.action_dim = action_space.high.item() if self.discrete else action_space.shape[-1]
        self.head_num = 1 if self.discrete else 2
        
        self.percentile = Percentile()
        self.symlogtwohot = SymLogTwoHot(255, -20, 20)
        
        kwargs = dict(in_features=feat_dim, hidden_features=hidden_dim, 
                      layers=layers-1, act=act, norm=norm, init_key=init_key)
        self.actor = MLPHead(**kwargs,out_features=self.action_dim,head_nums=self.head_num)
        self.critic = MLPHead(**kwargs,out_features=255)
        self.slow_critic = deepcopy(self.critic)
        
        # self.optimizer = nnx.Optimizer(self, optax.adam(1e-4))
        # optimizer = make_opt(lr=3e-5)
        # optimizer = make_simple_opt(lr=3e-5)
        # self.optimizer = nnx.Optimizer(self,optimizer)
    
    def get_logits(self, feats:jax.Array):
        actor_logits = self.actor(feats)
        critic_logits = self.critic(feats)
        return actor_logits, *critic_logits
    
    def value_loss(self,target:jax.Array,logits:jax.Array):
        return self.symlogtwohot.compute_loss(target, logits)
    
    def slow_update(self, decay:float=0.98):
        new_state = sg(nnx.state(self.critic))
        old_state = sg(nnx.state(self.slow_critic))
        ema_state = jax.tree.map(lambda x,y:(1-decay)*x+decay*y,new_state,old_state)
        nnx.update(self.slow_critic, sg(ema_state))
    
    def value(self, feats:jax.Array, critic_type:str):
        critic = getattr(self,critic_type)
        logits = critic(feats)[0]
        weights = jax.nn.softmax(logits, axis=-1)
        pred_value = self.symlogtwohot.decode(weights)
        return pred_value
    
    def sample_policy(self, feats:jax.Array, key:jax.random.PRNGKey):
        new_key, sub_key = jax.random.split(key, 2)
        if self.discrete:
            logits = self.actor(sg(feats))
            dist = Categorical(logits, unimix=0.01)
            return sg(dist.sample(sub_key)), new_key
        else:
            mean_std = self.actor(sg(feats))
            dist = bounded_normal(minstd=0.1,maxstd=1.,mean_std=mean_std)
            return sg(dist.sample(sub_key)), new_key
    
    @staticmethod
    def loss_fn(model, feats:jax.Array, actions:jax.Array, rewards:jax.Array, termination:jax.Array):
        assert feats.shape[1] - actions.shape[1] == 1
        actor_logits, critic_logits = model.get_logits(feats)
        skip_last = jax.tree.map(lambda x:x[:,:-1],actor_logits)
        dist = Categorical(*skip_last, unimix=0.01) if model.discrete else bounded_normal(0.1, 1., skip_last)
        log_probs = dist.logp(actions)
        entropy = dist.entropy()
        
        slow_value = model.value(feats,'slow_critic')
        value = model.symlogtwohot.decode(jax.nn.softmax(critic_logits))
        slow_lmbda_return = calc_lambda_return(rewards,slow_value,termination,model.gamma,model.lmbda)
        lmbda_return = calc_lambda_return(rewards,value,termination,model.gamma,model.lmbda)
        
        value_loss = model.value_loss(lmbda_return, critic_logits[:,:-1])
        slow_value_regularization_loss = model.value_loss(slow_lmbda_return, critic_logits[:,:-1])
        
        norm_ratio = model.percentile(lmbda_return, update=True)
        norm_advantage = (lmbda_return - value[:,:-1]) / norm_ratio
        policy_loss = -(log_probs*sg(norm_advantage)).mean()
        
        entropy_loss = entropy.mean()
        loss = policy_loss + value_loss + slow_value_regularization_loss - model.entropy_coef*entropy_loss
        
        rewards_sum_mean = jnp.mean(jnp.sum(rewards, axis=-1))
        rewards_mean = jnp.mean(rewards)
        metrics = {'Agent/policy_loss':policy_loss,
                   'Agent/value_loss':value_loss,
                   'Agent/entropy_loss':entropy_loss,
                   'Agent/norm_ratio':norm_ratio,
                   'Agent/reward_mean':rewards_mean,
                   'Agent/return_mean':rewards_sum_mean}
        return loss, metrics    
    
    @nnx.jit
    def update(self, optimizer:nnx.Optimizer, feats:jax.Array, actions:jax.Array, rewards:jax.Array, termination:jax.Array):
        (loss, metrics),grad = nnx.value_and_grad(self.loss_fn,has_aux=True)(self,feats,actions,rewards,termination)
        optimizer.update(grad)
        global_norm = optax.global_norm(grad)
        self.slow_update()
        metrics['Agent/global_norm'] = global_norm
        return loss, metrics  
    
    def save(self, save_path:str):
        _, state = nnx.split(self)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path,'agent')
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(save_path, state)
             
          
if __name__ == '__main__':
    # from tqdm import tqdm
    
    # action_space = Space(jnp.uint8,(),0,18)
    # init_key = nnx.Rngs(0)
    # Agent = ActorCritic(1024,512,2,action_space,0.97,0.5,1e-4,'silu','LayerNorm',init_key)
    # feats = jax.random.normal(init_key.noise(),(1024,17,1024))
    # action = jax.random.randint(init_key.noise(),(1024,16),0,10)
    # reward = jax.random.normal(init_key.noise(),(1024,16))
    # termination = jax.random.normal(init_key.noise(),(1024,16)) > -1.
    # termination = jnp.float32(termination)
    # action,key=Agent.sample_policy(feats[:,0],init_key.noise())
    # print(action.squeeze(0))
    # print(feats[:,0].shape)
    
    
    # with tqdm(total=1000,desc='Epoch') as pbar:
    #     for epoch in range(1000):
    #         loss, metrics = Agent.update(feats,action,reward,termination)
    #         pbar.set_postfix({'loss':loss,'global_norm':metrics['Agent/global_norm']})
    #         pbar.update(1)
    
    # value = jnp.arange(0,64).reshape(4,16)
    # reward = jnp.ones_like(value)
    # termination = jnp.zeros_like(value)
    # lambda_return = calc_lambda_return(reward,value,termination,0.98,1)
    # print(value)
    # print(lambda_return)
    
    pass
