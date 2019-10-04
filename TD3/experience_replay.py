import random
import numpy as np

# 1 - initialize the Experience replay memory
class ReplayBuffer(object):
    
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)
    
    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size = batch_size)
        batch_state, batch_next_state, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_state.append(np.array(state,copy=False))
            batch_next_state.append(np.array(next_state,copy=False))
            batch_actions.append(np.array(action,copy=False))
            batch_rewards.append(np.array(reward,copy=False))
            batch_dones.append(np.array(done,copy=False))
        return np.array(batch_state), np.array(batch_next_state), np.array(batch_actions), np.array(batch_rewards).reshape(-1,1), np.array(batch_dones).reshape(-1,1)
    









