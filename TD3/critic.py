import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 3 - two neural networks for the critic model and two neural networks for the critic target
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        # Defining the 1st critic neural network
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        # Defining the 2nd critic neural network
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        # forward bropagation on 1st critic
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # forward bropagation on 2nd critic 
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2
    # to be used for gradient acent to update the weights
    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        # forward bropagation on 1st critic
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

    

    









