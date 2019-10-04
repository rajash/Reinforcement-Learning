import numpy as np
import matplotlib.pyplot as plt
import random

N = 1000
d  = 9

conversion_rate = [ 0.05, 0.13, 0.09, 0.16, 0.11 ,0.04, 0.2, 0.08, 0.01]
X = np.array(np.zeros([N, d]))
for i in range(N):
    for j in range (d):
        if np.random.rand() <= conversion_rate[j]:
            X[i, j] = 1

strategies_selected_rs = []
strategies_selected_ts = []
total_reward_rs = 0
total_reward_ts = 0
numbers_of_rewards_1 =  [0] * d
numbers_of_rewards_0 =  [0] * d 

for n in range(N):
    # random strategy
    strategy_rs = random.randrange(d)
    strategies_selected_rs.append(strategy_rs)
    reward_rs = X[n, strategy_rs]
    total_reward_rs += reward_rs
    # Thampson sampling
    strategy_ts = 0
    max_random = 0
    for i in range(d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1,
                                         numbers_of_rewards_0[i] + 1)
        if (random_beta > max_random):
            max_random = random_beta
            strategy_ts = i
    reward_ts = X[n, strategy_ts]
    if reward_ts == 1:
        numbers_of_rewards_1[strategy_ts] += 1
    else:
        numbers_of_rewards_0[strategy_ts] += 1
    strategies_selected_ts.append(strategy_ts)
    total_reward_ts += reward_ts

absolute_return = (total_reward_ts - total_reward_rs) * 100
relative_return = (total_reward_ts - total_reward_rs)/total_reward_rs *100
print('Absolute Retrun {:.0f} $'.format(absolute_return))
print('Relative Retrun {:.0f} %'.format(relative_return))

plt.hist(strategies_selected_ts)
plt.title('Histogram of Selections')
plt.xlabel('Strategy')
plt.ylabel('Number of Times The Strategy was Selected')
plt.show()

rewards_strategies = [0] * d
for n in range(N):
     # Best Strategy
     for i in range(0, d):
         rewards_strategies[i] = rewards_strategies[i] + X[n, i]     
         total_reward_bs = max(rewards_strategies)

print('Best Strategy: ',total_reward_bs)

strategies_selected_ts = []
total_reward_ts = 0
total_reward_bs = 0 
numbers_of_rewards_1 = [0] * d 
numbers_of_rewards_0 = [0] * d 
rewards_strategies = [0] * d 
regret = [] 
for n in range(0, N):
     # Thompson Sampling
     strategy_ts = 0
     max_random = 0
     for i in range(0, d):
         random_beta = random.betavariate(numbers_of_rewards_1[i] + 1,
                                          numbers_of_rewards_0[i] + 1)
         if random_beta > max_random:
             max_random = random_beta
             strategy_ts = i
     reward_ts = X[n, strategy_ts]
     if reward_ts == 1:
         numbers_of_rewards_1[strategy_ts] = numbers_of_rewards_1[strategy_ts] + 1
     else:
         numbers_of_rewards_0[strategy_ts] = numbers_of_rewards_0[strategy_ts] + 1
     strategies_selected_ts.append(strategy_ts)
     total_reward_ts = total_reward_ts + reward_ts
     # Best Strategy
     for i in range(0, d):
         rewards_strategies[i] = rewards_strategies[i] + X[n, i]     
         total_reward_bs = max(rewards_strategies)
     # Regret
     regret.append(total_reward_bs - total_reward_ts)

# And same, the regret of the Random Strategy is simply computed as the difference between the best strategy and the random selection algorithm:

# Regret of the Random Strategy
strategies_selected_rs = [] 
total_reward_rs = 0 
total_reward_bs = 0 
numbers_of_rewards_1 = [0] * d 
numbers_of_rewards_0 = [0] * d 
rewards_strategies = [0] * d 
regret = [] 
for n in range(0, N):
     # Random Strategy
     strategy_rs = random.randrange(d)
     strategies_selected_rs.append(strategy_rs)
     reward_rs = X[n, strategy_rs]
     total_reward_rs = total_reward_rs + reward_rs
     # Best Strategy
     for i in range(0, d):
         rewards_strategies[i] = rewards_strategies[i] + X[n, i]     
         total_reward_bs = max(rewards_strategies)
     # Regret
     regret.append(total_reward_bs - total_reward_rs)

# Plotting the Regret Curve 
plt.plot(regret) 
plt.title('Regret Curve') 
plt.xlabel('Round') 
plt.ylabel('Regret') 
plt.show()