from typing import Dict

import tensorboard
from tensorboardX import SummaryWriter
import gymnasium as gym
from einops import rearrange
import imageio

class Logger:
    def __init__(self,log_path:str):
        self.writer = SummaryWriter(log_path)
        self.step = {}
    
    def log_dict(self,metrics:Dict):
        for key in metrics:
            if key not in self.step:
                self.step[key]=0
            else:
                self.step[key]+=1
            
            # if 'Image' in key:
            #     self.writer.add_image(key,metrics[key],self.step[key])
            if 'video' in key:
                self.writer.add_video(key,metrics[key],self.step[key],fps=15)
            else:
                self.writer.add_scalar(key,metrics[key],self.step[key])

if __name__ == '__main__':
    # import numpy as np

    # logger = Logger('./logdir')
    # env = gym.make('ALE/Pong-v5')
    # obs,_ = env.reset(seed=5)
    # video = []
    # for _ in range(100):
    #     video.append(obs)
    #     obs,_,_,_,_ = env.step(env.action_space.sample())

    # obs = rearrange(obs,'H W C -> C H W')/255.
    # imageio.mimsave('res.gif',video,fps=20)
    # metrics = {'worldmodel/rep_loss':0.01,'worldmodel/dyn_loss':0.02,'play/image':obs}
    # logger.log_dict(metrics)
    pass
