import gym
import numpy as np 

env = gym.make('Taxi-v2')
env.reset()
env.render()

num_actions = env.action_space.n
num_states = env.observation_space.n

Q = np.zeros([num_states, num_actions])

gamma = 0.9
alpha = 0.9

# # train 
for ith_episode in range(1,1001):
    done = False
    rewards_accu = 0
    state = env.reset()
    while done != True:
        action = np.argmax(Q[state])
        next_state, reward, done, _ = env.step(action)
        TD =  reward + gamma * np.max(Q[next_state,]) - Q[state, action]
        Q[state, action] += alpha * TD

        rewards_accu += reward

        state = next_state

    if ith_episode % 100 == 0:
        print('Episode {} Total Reward: {}'.format(ith_episode, rewards_accu))
        
# test
done = False
rewards_accu = 0
state = env.reset()
env.render()
while done != True:
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    env.render()
    rewards_accu += reward

print("Reward: %r" % rewards_accu) 
