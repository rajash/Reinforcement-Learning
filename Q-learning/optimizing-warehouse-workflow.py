import numpy as np
import string

print("____________\n|A  B  C | D\n|__   |G   H|\n E |F |__   |\n|I  J  K   L|\n|_____    __|")

start = input('Start: ').upper()
end = input('End: ').upper()
iterations = int(input('#Iterations: '))
print(start,end)
gamma = 0.75
alpha = 0.9

location_to_state = dict()
for i,loc in enumerate(list(string.ascii_uppercase[0:12])):
    location_to_state[loc] = i

actions = list(location_to_state.values())

R = np.array( [
#    A, B, C, D, E, F, G, H, I, J, K, L
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # A
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0], # B
    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # C
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], # D
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # E
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # F
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], # G
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1], # H
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], # I
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0], # J
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], # K
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]  # L
])

def Q_learning(R, iterations):
    Q = np.array(np.zeros([12, 12]))
    for i in range(iterations):
        current_state = np.random.randint(0,12)
        playable_actions = []
        for j in range(12):
            if R[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] += alpha * TD
    return Q

def route(starting_location, ending_location, iterations):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 1000
    opt_route = [starting_location]
    next_location = starting_location
    Q = Q_learning(R_new, iterations)
    while (next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = next((loc for loc, state in location_to_state.items() if next_state == state), None) 
        opt_route.append(next_location)
        starting_location = next_location
    return opt_route

print('Optimal Route:',route(start, end, iterations))
