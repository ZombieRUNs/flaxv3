from omegaconf import DictConfig
import os
from copy import deepcopy

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from elements import Space
from tqdm import tqdm
import einops
import pandas as pd

from networks.dreamerv3 import Dreamerv3
from networks.agent import ActorCritic
from utils.utils import Logger
from utils.env_wrappers import build_eval_env,build_eval_vec_env
import hydra
import orbax.checkpoint as ocp

sg = jax.lax.stop_gradient

def restore_model(model,model_type,ckpt_dir,config,model_name):
    checkpointer = ocp.StandardCheckpointer()
    abstract_model = nnx.eval_shape(lambda: model_type(**config))
    graphdef, abstract_state = nnx.split(abstract_model)
    state_restored = checkpointer.restore(os.path.join(ckpt_dir,model_name), abstract_state)
    model = nnx.merge(graphdef, state_restored)
    return model

@hydra.main(config_path='./config',config_name='config')
def main(config:DictConfig):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.eval.device)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    config.agent.feat_dim = config.dreamerv3.rssm.stoch*config.dreamerv3.rssm.stoch+config.dreamerv3.rssm.deter
    ckpt_path = config.eval.ckpt_path
 
    dumy_env = build_eval_env(config.eval.env_name,(64,64))
    action_space = Space(np.uint8,(),0,dumy_env.action_space.n)
    
    env = build_eval_vec_env(env_name=config.eval.env_name,num_envs=config.eval.num_envs,image_size=(64,64))
    init_key = nnx.Rngs(config.training.seed)
    agent_config = dict(**config.agent,action_space=action_space,init_key=init_key)
    dreamerv3_config = dict(enc_cfg=getattr(config,'dreamerv3').encoder,
                          seq_cfg=getattr(config,'dreamerv3').rssm,
                          dec_cfg=getattr(config,'dreamerv3').decoder,
                          action_space=action_space,init_key=init_key)
    
    agent = ActorCritic(**agent_config)
    dreamerv3 = Dreamerv3(**dreamerv3_config)
    
    agent = restore_model(agent,ActorCritic,ckpt_path,agent_config,'agent')
    dreamerv3 = restore_model(dreamerv3,Dreamerv3,ckpt_path,dreamerv3_config,'worldmodel')
    
    enc_for_eval = deepcopy(dreamerv3.enc)
    seq_for_eval = deepcopy(dreamerv3.seq)
        
    @nnx.jit
    def sample_policy(carry,obs,action,reset):
        tokens = enc_for_eval(obs)
        tokens = einops.rearrange(tokens, '... H W C -> ... (H W C)')
        carry, _ = seq_for_eval._observe(carry, tokens, action, reset)
        # carry,_ = sg(dreamerv3.encode(carry,obs,action,reset))
        feats = jnp.concatenate([carry['deter'],carry['stoch']],axis=-1)
        action, carry['key'] = sg(agent.sample_policy(feats,carry['key']))
        return carry, action
    
    obs, _ = env.reset()
    carry = dreamerv3.init_carry(config.eval.num_envs, jax.random.PRNGKey(0))
    sum_rewards = np.zeros(config.eval.num_envs)
    action = env.action_space.sample()
    reset = np.array([False for _ in range(config.eval.num_envs)])
    episodes = 0
    episodes_return = []
    
    with tqdm(total=config.eval.episodes,desc='Epoch',ncols=100) as pbar:
        while True:
            # carry,_ = sg(dreamerv3.encode(carry,obs,action,reset))
            # feats = jnp.concatenate([carry['deter'],carry['stoch']],axis=-1)
            # action, carry['key'] = sg(agent.sample_policy(feats,carry['key']))
            # action = np.asarray(action)
            carry, action = sample_policy(carry, obs, action, reset)
            action = np.asarray(action)

            next_obs, reward, termination, truncated, info = env.step(action)
            reset = np.logical_or(termination, info["life_loss"])
            
            done_flags = np.logical_or(termination,truncated)
            if done_flags.any():
                for i in range(config.eval.num_envs):
                    if done_flags[i]:
                        episodes += 1
                        episodes_return.append(sum_rewards[i])  
                        pbar.set_postfix({'episode':episodes,'return':sum_rewards[i]})
                        pbar.update(1)
                        sum_rewards[i] = 0
            
            sum_rewards += reward
            obs = next_obs
            
            if episodes == config.eval.episodes:
                break
    
    print('Everage Return:', sum(episodes_return)/config.eval.episodes)
    print('EveryEpisode:', episodes_return)
    
    csv = {'Checkpoint':[100000],'Everage':[sum(episodes_return)/config.eval.episodes]}
    csv = pd.DataFrame(csv)
    csv.to_csv('test.csv',index=False)
    
if __name__ == '__main__':
    main()