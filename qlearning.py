import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import aisd_examples
import time
import os

# Create the environment
env = gym.make("aisd_examples/CreateRedBall-v0", render_mode="human")

# Get the number of states from the environment's observation space
numstates = env.observation_space.n

# Initialize the Q-table with random values
qtable = np.random.rand(numstates, env.action_space.n)

# Hyperparameters
episodes = 100
gamma = 0.1
epsilon = 0.08
original_epsilon = epsilon
decay = 0.1
rewards = []
ep_steps = []

# Training loop
for i in range(episodes):
    state, info = env.reset()
    steps = 0
    total_reward = 0

    done = False
    while not done:
        os.system('clear')
        print("Episode #", i+1, "/", episodes)
        print('steps:',steps)
        env.render()
        time.sleep(0.05)

        steps += 1

        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qtable[state])

        next_state, reward, done, _, _ = env.step(action)

        qtable[state][action] = reward + gamma * max(qtable[next_state])

        state = next_state
        total_reward += reward

    print('Total reward:', total_reward)
    rewards.append(total_reward)
    ep_steps.append(steps)

    epsilon -= decay * epsilon

    print("\nDone in", steps, "steps")
    print("Total Accumulated Reward:", total_reward)
    time.sleep(0.8)

env.close()

# Plotting

plt.title('Original Hyperparameters')
plt.xlabel('Episodes')
plt.ylabel('Sum of Rewards during episode')

hyperparameters_text = 'gamma = {:.3f}, epsilon = {:.3f}, decay = {:.3f}'.format(gamma, original_epsilon, decay)
plt.text(0.95, 0.05, hyperparameters_text, ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

plt.plot(rewards)
plt.legend(['Sum of Rewards'])
plt.show()

