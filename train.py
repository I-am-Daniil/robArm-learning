import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# Initialize environment with parallel workers and normalization
env = make_vec_env("PandaPickAndPlace-v1", n_envs=4)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Neural network architecture for the policy
policy_kwargs = dict(net_arch=[512, 512, 256])

# PPO model configuration
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01
)

# Training and saving
model.learn(total_timesteps=1_000_000)
model.save("robot_arm_model")
env.save("vec_normalize.pkl")