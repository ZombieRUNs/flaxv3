from dataclasses import dataclass
from elements import Space
from typing import Dict
from collections import deque
from copy import deepcopy

import numpy as np
import jax.numpy as jnp
import jax

def softmax(x, temperature, axis):
    x = -x/temperature
    x -= np.max(x, axis=axis, keepdims=True)
    x = np.exp(x)/np.sum(np.exp(x),axis=-1,keepdims=True)
    return x

class Replaybuffer:
    def __init__(self,obs_space:Dict,capacity:int,num_envs:int,action_space:Space,warmup_length:int,sample_length:int):
        assert capacity%num_envs == 0,(capacity,num_envs)
        single_length = capacity//num_envs
        self.obs_buffer = {}
        for key,value in obs_space.items():
            self.obs_buffer[key] = np.empty([single_length,num_envs,*value.shape],dtype=value.dtype)
        self.action_buffer = np.empty([single_length,num_envs,*action_space.shape],dtype=action_space.dtype)
        self.reward_buffer = np.empty([single_length,num_envs],dtype=np.float32)
        self.termination_buffer = np.empty([single_length,num_envs],dtype=np.bool_)
        self.visit_counts = np.zeros((single_length, num_envs), dtype=np.int64)
        self.indexs = np.stack([np.arange(0, capacity) for _ in range(num_envs)],axis=1)

        self.last_pointer = -1
        self.length = 0
        self.capacity = single_length
        self.num_envs = num_envs
        self.warmup_length = warmup_length
        self.sample_length = sample_length
    
    @property
    def ready(self):
        return self.length*self.num_envs>self.warmup_length and self.length>self.sample_length
    
    def __len__(self):
        return self.length*self.num_envs

    def add(self,step:Dict):
        self.last_pointer = (self.last_pointer+1)%self.capacity
        for key in step:
            buffer = getattr(self,f'{key}_buffer')
            if key.startswith('obs'):
                for sub_key in buffer:
                    buffer[sub_key][self.last_pointer] = step[key][sub_key]
            else:
                buffer[self.last_pointer] = step[key]
        if self.length<self.capacity:
            self.length+=1
        else:
            self.visit_counts[self.last_pointer] = 0
    
    def _compute_visit_probs(self, n, env_id, temperature):
        logits = np.float32(self.visit_counts[:n, env_id])
        probs = softmax(logits, temperature, 0)
        return probs
    
    def sample_idx(self, env_id, batch_size):
        assert self.length >= self.sample_length
        n = self.length - self.sample_length + 1
        probs = self._compute_visit_probs(n, env_id, 20.)
        start_idx = np.random.choice(self.indexs[:n, env_id],(batch_size//self.num_envs,), p=probs)
        value,new_count = np.unique(start_idx, return_counts=True)
        self.visit_counts[value, env_id] += new_count
        return start_idx
    
    def sample(self,batch_size:int,normalized:bool=False):
        batch_length = self.sample_length
        if batch_size < self.num_envs:
            batch_size = self.num_envs
        obs,action,reward,termination = {key:[] for key in self.obs_buffer},[],[],[]
        for env_id in range(self.num_envs):
            if not normalized:
                indexs = np.random.randint(0,self.length-batch_length+1,(batch_size//self.num_envs,))
            else:
                indexs = self.sample_idx(env_id, batch_size)
            for key in self.obs_buffer:
                obs[key].append(np.stack([self.obs_buffer[key][index:index+batch_length,env_id] for index in indexs]))
            action.append(np.stack([self.action_buffer[index:index+batch_length,env_id] for index in indexs]))
            reward.append(np.stack([self.reward_buffer[index:index+batch_length,env_id] for index in indexs]))
            termination.append(np.stack([self.termination_buffer[index:index+batch_length,env_id] for index in indexs]))
        step = {'obs':{key: np.concatenate(obs[key],axis=0)[:,1:] for key in obs},
                'action':np.concatenate(action,axis=0)[:,:-1],
                'reward':np.concatenate(reward,axis=0)[:,:-1],
                'termination':np.concatenate(termination,axis=0)[:,:-1]}
        return step

class Episode:
    def  __init__(self, obs_space:Dict):
        self.obs_buff = {key:[] for key in obs_space}
        self.action_buff = []
        self.reward_buff = []
        self.termination_buff = []
    
    def __len__(self):
        return len(self.action_buff)

class Datasets:
    def __init__(self, num_envs:int, obs_space:Dict, action_space:Space, max_episodes:int, max_length:int, warmup_length:int,sample_length:int):
        self.episodes = deque(maxlen=max_episodes)
        self.init_episode = lambda : Episode(obs_space)
        self.current_episode = [self.init_episode() for _ in range(num_envs)]
        self.length = 0
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.sample_length = sample_length
        self.num_envs = num_envs
        self.obs_space = obs_space
        self.action_space = action_space
    
    def add(self, step:Dict):
        for key, value in step.items():
            for env_id in range(self.num_envs):
                buffer = getattr(self.current_episode[env_id], f'{key}_buff')
                if key.startswith('obs'):
                    for sub_key in buffer:
                        buffer[sub_key].append(value[sub_key][env_id]) 
                else:
                    buffer.append(value[env_id])
                if key == 'termination' and value.any():
                    self.episodes.append(deepcopy(self.current_episode[env_id]))
                    self.current_episode[env_id] = self.init_episode()
        
        self.length += self.num_envs
        if self.length > self.max_length:
            self.length -= len(self.episodes[0])
            self.episodes.popleft()
    
    def init_samples(self, batch_sizes:int):
        obs = {key: np.empty([batch_sizes,self.sample_length,*value.shape],dtype=value.dtype) 
               for key, value in self.obs_space.items()}
        action = np.empty([batch_sizes,self.sample_length,*self.action_space.shape],dtype=self.action_space.dtype)
        reward = np.empty([batch_sizes,self.sample_length],dtype=np.float32)
        termination = np.empty([batch_sizes,self.sample_length],dtype=np.bool_)
        init_samples = {'obs':obs, 'action':action, 'reward':reward, 'termination':termination}
        return init_samples
    
    @property
    def ready(self):
        return self.length>=self.warmup_length and len(self.episodes)>0
    
    def sample(self, batch_size:int, *args):
        assert self.ready
        # samples = {'obs':{key:[] for key in self.obs_space},'action':[],'reward':[],'termination':[]}
        samples = self.init_samples(batch_size)
        episode_length = np.asarray([len(episode) for episode in self.episodes])
        p = episode_length/sum(episode_length)
        episode_indexes = np.random.choice(len(episode_length),batch_size,p=p)
        for i,episode_index in enumerate(episode_indexes):
            episode = self.episodes[episode_index]
            sample_index = np.random.randint(0,max(len(episode)-self.sample_length+1,0))
            for key in samples:
                buffer = getattr(episode, f'{key}_buff')
                if key.startswith('obs'):
                    for sub_key in buffer:
                        samples[key][sub_key][i] = np.stack(buffer[sub_key][sample_index:sample_index+self.sample_length],axis=0)
                else:
                    samples[key][i] = np.stack(buffer[sample_index:sample_index+self.sample_length],axis=0)
        
        for key in samples:
            if key.startswith('obs'):
                for sub_key in samples['obs']:
                    samples['obs'][sub_key] = samples['obs'][sub_key][:,1:]
            else:
                samples[key] = samples[key][:,:-1]
        
        return samples
            

if __name__ == '__main__':
    # from tqdm import tqdm
    # obs_space = {'image':Space(dtype=np.uint8,shape=(64,64,3),low=0,high=255),}
    # action_space = Space(dtype=np.float32,shape=(),low=0,high=18)
    # rpbf = Replaybuffer(obs_space=obs_space,
    #                     capacity=int(1e6),
    #                     num_envs=1,
    #                     warmup_length=4096,
    #                     sample_length=64,
    #                     action_space=action_space)
    
    # for _ in range(256):
    #     step = {'obs':{'image':np.random.randint(0,255,(1,64,64,3)),},
    #             'action':np.random.randint(0,18,(1,)),
    #             'reward':np.random.randn(1,),
    #             'termination':np.random.randn(1,)}
    #     rpbf.add(step)   

    # with tqdm(total=99000,ncols=150) as pbar:
    #     for _ in range(99000):
    #         step = {'obs':{'image':np.random.randint(0,255,(1,64,64,3)),},
    #         'action':np.random.randint(0,18,(1,)),
    #         'reward':np.random.randn(1,),
    #         'termination':np.random.randn(1,)}
    #         rpbf.add(step) 
    #         sam = rpbf.sample(16)
    #         pbar.update(1)
    #     print('mean',rpbf.visit_counts[50000:90000].mean())
    #     print('max',rpbf.visit_counts[50000:90000].max())
    #     print('min',rpbf.visit_counts[50000:90000].min())
    
    from tqdm import tqdm
    obs_space = {'image':Space(dtype=np.uint8,shape=(64,64,3),low=0,high=255),}
    action_space = Space(dtype=np.float32,shape=(),low=0,high=18)
    datasets = Datasets(1,obs_space,action_space,200,int(1E5),5000,65)
    
    with tqdm(total=100000) as pbar:
        for epoch in range(100000):
            termination = (np.random.randint(0,5,1) < -1)
            if (epoch+1) % 128 == 0:
                termination = (np.random.randint(0,5,1) > -1)
            step = {'obs':{'image':np.random.randint(0,255,(1,64,64,3))},
                    'action':np.random.randint(0,18,(1,)),
                    'reward':np.random.randn(1,),
                    'termination':termination}
            datasets.add(step)
        
            if datasets.ready:
                sample=datasets.sample(batch_size=16)
            pbar.update(1)
            pbar.set_postfix({'length':datasets.length,'episodes':len(datasets.episodes)})
    
