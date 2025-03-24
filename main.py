import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import time

# Load environment with normalization stats
env = DummyVecEnv([lambda: gym.make("PandaPickAndPlace-v1")])
env = VecNormalize.load("vec_normalize.pkl", env)
env.training = False

# Load trained model
model = PPO.load("robot_arm_model", env=env)

# Run inference loop
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, _, done, _ = env.step(action)
    env.render()
    time.sleep(1/60.)
    if done:
        obs = env.reset()