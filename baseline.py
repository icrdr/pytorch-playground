# %%
import gym

# from stable_baselines3 import DQN
# from stable_baselines3.dqn import MlpPolicy
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.cmd_util import make_vec_env

# Parallel environments
env = gym.make('LunarLander-v2')
model = A2C(MlpPolicy, env, verbose=1)

# env = gym.make('LunarLander-v2')
# model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=500000, log_interval=1000)


# %%
import numpy as np
for i in range(10):
    done = False
    score = 0
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        score += reward
        env.render()
    print(score)
# %%
