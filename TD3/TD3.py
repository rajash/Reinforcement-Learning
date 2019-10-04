import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experience_replay import ReplayBuffer
from actor import Actor
from critic import Critic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TD3(object):
# Twin Delay DDGP
    def __init__(self,  state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
    
    def select_action(self, state):
        state = torch.Tensor(state.reshape(1,-1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size = 100, discount=0.99, tau=0.005, policy_noise= 0.2, noise_clip=0.5, policy_freq=2):
        for it in range(iterations):
            # step 4: we sample batch of transitions (s, s', a, r) from the memory
            batch_state, batch_next_state, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state = torch.Tensor(batch_state).to(device)
            next_state = torch.Tensor(batch_next_state).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # step 5: from the next state s', the actor traget plays next action a'
            next_action = self.actor_target(next_state)

            # step 6: we add Gaussian noise to the next action a' and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # step 7: the two critic targets take each the couple (s', a') as input and return  2 Q-values  Qt1(s', a') and Qt2(s', a') as output
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # step 8: we keep the minimum of these Q-values min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)

            # step 9: we get the final target of the 2 critic models, which is : Qt = r + gamma * target_Q
            target_Q = reward + (1 - done) * discount * target_Q

            # step 10: the 2 critic models take each the couple (s, a) as input and return 2 Q-values Qt1(s, a) and Qt2(s, a) as outputs
            current_Q1, current_Q2 = self.critic(state, action)

            # step 11: we compute the less coming from the 2 critic models : critic loss = mse_loss(Q1(s,a), Qt) + mse_loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # step 12:  we backpropagate the critic loss and update the parameters of the 2 critic models with SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # step 13: one every 2 iterations, we update our Actor model by performing gradient ascent on the output of the first critic model
            if it % policy_freq == 0:
                # deterministic policy gradient DPG
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                self.actor.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Delay 
                # step 14: still once every 2 iterations, we update the weights of the actor target by polyak averaging 
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                # step 15: still ones every 2 iterations, we uodate the weights of the critic target by polyak averaging
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
    
    def load(self, filename,  directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

    
