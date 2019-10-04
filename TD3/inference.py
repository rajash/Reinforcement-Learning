import time
import os
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
from gym import wrappers
from torch.autograd import Variable
from collections import deque
from experience_replay import ReplayBuffer
from TD3 import TD3

env_name = 'HalfCheatahBulletEnv-v0'
# env_name = 'Walker2DBulletEnv-v0'
# env_name = 'AntBulletEnv-v0'

seed = 0

#file name for the two saved models: the Actor and Critic models
filename = '%s_%s_%s' % ('TD3', env_name, str(seed))
print('----------------------------------------')
print('Settings %s' % (filename))
print('----------------------------------------')


# create pyBullet enviroment
env = gym.make(env_name)

# create new folder dir. in which the final results will be populated
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

eval_epispdes = 10
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')
max_episode_steps = env._max_episode_steps
save_env_vid = True
if save_env_vid:
    env = wrappers.Monitor(env, monitor_dir, force=True)
    env.reset()

# set seeds and get the necessary info on the states and actions in the chosen environment
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# create the policy network (Actor model)
policy = TD3(state_dim, action_dim, max_action)
policy.load(filename,'./pytorch_models')


# evaluation resuls over 10 episodes are stored
def evaluate_policy(policy, eval_episodes = 10):
    avg_reward = 0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print('----------------------------------------')
    print('Average Reward Over The Evaluation Step %f' % (avg_reward))
    print('----------------------------------------')
    return avg_reward

_ = evaluate_policy(policy, eval_episodes = eval_epispdes)




