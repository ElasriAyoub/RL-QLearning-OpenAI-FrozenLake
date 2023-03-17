import FrozenLake as fl

# ENVIRONNEMENT
env=fl.MyFrozenLakeEnv()
env.reset() #Initialization
env.render() #Display
print("")
# ACTIONS POSSIBLES
actions={'Left':0,'Down':1,'Right':2,'Up':3}
chaine=""
scenario = (2*['Right'] + 3*['Down'] + ['Right'])
episode_states, episode_actions, episode_rewards = [],[],[]

# AGENT : FAIT DES ACTIONS
for action in scenario:
    episode_states += [env.s]
    episode_actions += [actions[action]]
    new_state, reward, done, _ = env.step(actions[action])
    episode_rewards+=[reward]
    for i in range(len(episode_states)):
        chaine = "(s=" + str(episode_states[i]) + ",a=" + str(episode_actions[i]) + ")->r=" + str(episode_rewards[i])
    print(chaine)
    env.render()
    print("")

chaine=""
for i in range (len(episode_states)):
    chaine+= "(s=" +str(episode_states[i])+",a=" + str(episode_actions[i]) + ")->r=" + str(episode_rewards[i])+"->"
print(chaine)

