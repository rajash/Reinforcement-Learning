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

# env_name = 'HalfCheetahBulletEnv-v0'
env_name = 'AntBulletEnv-v0'

seed = 0
start_timesteps = 1e4
eval_freq = 5e3
max_timesteps = 1e6
save_models = True
expl_noise = 0.1
batch_size = 100
discount = 0.99
tua = 0.005
policy_noise = 0.2
noise_clip = 0.5
policy_freq = 2

#file name for the two saved models: the Actor and Critic models
filename = '%s_%s_%s' % ('TD3', env_name, str(seed))
print('----------------------------------------')
print('Settings %s' % (filename))
print('----------------------------------------')

# create a folder inside which will be saved the trained models
if not os.path.exists('./results'):
    os.makedirs('./results')
if save_models and not os.path.exists('./pytorch_models'):
    os.makedirs('./pytorch_models')

# create pyBullet enviroment
env = gym.make(env_name)

# set seeds and get the necessary info on the states and actions in the chosen environment
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# create the policy network (Actor model)
policy = TD3(state_dim, action_dim, max_action)

# create the experience replay memo
replay_buffer = ReplayBuffer()

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
evaluations = [evaluate_policy(policy)]

# create new folder dir. in which the final results will be populated
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')
max_episode_steps = env._max_episode_steps
save_env_vid = False
if save_env_vid:
    env = wrappers.Monitor(env, monitor_dir, force=True)
    env.reset()

# initialize variables
total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
t0 = time.time()

# Training
while total_timesteps < max_timesteps:
    if done:
        if total_timesteps != 0:
            print('Total Timesteps: {} Episode Num: {} Reward: {}'.format(total_timesteps, episode_num, episode_reward))
            policy.train(replay_buffer, episode_timesteps, batch_size, discount, tua, policy_noise, noise_clip, policy_freq)

        if timesteps_since_eval >= eval_freq:
            timesteps_since_eval %= eval_freq
            evaluations.append(evaluate_policy(policy))
            policy.save(filename, directory='./pytorch_models')
            np.save('./results/%s' % (filename), evaluations)
    
        obs = env.reset()

        done = False

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # before 10000 timesteps we play random actions
    if total_timesteps < start_timesteps:
        action = env.action_space.sample()
    else:
        action = policy.select_action(np.array(obs))

        if expl_noise != 0:
            action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)
    
    new_obs, reward, done, _ = env.step(action)

    done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)

    episode_reward += reward

    replay_buffer.add((obs, new_obs, action, reward, done_bool))

    obs = new_obs
    episode_timesteps += 1
    total_timesteps +=1
    timesteps_since_eval += 1

evaluations.append(evaluate_policy(policy))
if save_models: 
    policy.save('%s' % (filename), directory='./pytorch_models')
np.save('./results/%s' % (filename), evaluations)





