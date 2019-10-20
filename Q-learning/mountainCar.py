import gym

env = gym.make('MountainCar-v0')

MAX_NUM_EPISODES = 5000

for episode in range(MAX_NUM_EPISODES):
    done = False
    obs = env.reset()
    total_reward = 0.0
    step = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        step +=1
        obs = next_state
        print("\n Episode #{} ended in {} steps.".format(episode, step+1))

env.close()
