import numpy as np
import FrozenLake as fl
#Environnement
env=fl.MyFrozenLakeEnv(rewards={'S':0,'F':0,'H':-1,'G':5,'NoChange':0})
#Initialization de la Q-Table
Q = np.zeros([env.observation_space.n,env.action_space.n])
#Learning
lr, y, num_episodes = .8, .95 , 2000

for i in range(num_episodes): #Pour chaque partie
    s = env.reset() #Reset environment and get first new observation
    d, j = False, 0  # d: True si partie terminated (hole: perdu ou goal: won)
    # The Q-Table learning algorithm
    while j < 99:
        j += 1
        # Select action
        a = env.action_space.sample()
        # Get new state and reward from environment
        s1, r, d, _ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr*(r+y*np.max(Q[s1,:]) - Q[s,a])
        # Update current state
        s = s1
        if d == True:  # Si on termine -> nouvel épisode
            break

print(" ")
print("Final Q-Table values : \n" , np.round(Q,5))

#USE OF QTABLE FOR OPTIMAL PLAY

print("Enchainement optimal (pi^*) des actions est: \n")
s=env.reset()
env.render()
print(" ")


while s<15:
    best_action = np.argmax(Q[s,:])
    snext, r, d, _ = env.step(best_action)
    env.render()
    s=snext
    print(" ")
