import gymnasium as gym
import aisd_examples
from stable_baselines3 import DQN

env = gym.make("aisd_examples/CreateRedBall-v0", render_mode="human")
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100, log_interval=4)
model.save("dqn_blocks")
del model # remove to demonstrate saving and loading

model = DQN.load("dqn_blocks")
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
