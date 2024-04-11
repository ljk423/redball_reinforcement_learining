import gymnasium as gym
import aisd_examples
import matplotlib.pyplot as plt
import numpy as np
import os

# Create the environment
env = gym.make("aisd_examples/CreateRedBall-v0", render_mode="human")

episode_returns = []

max_steps_per_episode = 100
max_episodes = 100

for episode in range(1, max_episodes + 1):
    observation, info = env.reset()
    episode_return = 0

    for step in range(max_steps_per_episode):
        os.system('clear')
        print("Episode #", episode, "/", max_episodes)
        print('steps:', step)

        error = observation - (env.observation_space.n // 2)

        angular_velocity = -0.01 * error

        action = int(angular_velocity * 320)

        observation, reward, done, _, info = env.step(action)
        episode_return += reward

        if done or step == max_steps_per_episode - 1:
            episode_returns.append(episode_return)
            break

# Plot the returns per episode
plt.plot(episode_returns)
plt.title('Returns per Episode')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.show()

# Close the environment
env.close()

