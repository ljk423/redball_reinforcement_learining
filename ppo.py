import gymnasium as gym
import aisd_examples
from stable_baselines3 import PPO

env = gym.make("aisd_examples/CreateRedBall-v0", render_mode="human")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100, log_interval=4)

model.save("ppo_cartpole")
del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
