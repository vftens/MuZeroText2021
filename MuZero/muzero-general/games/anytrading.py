# Применение библиотеки gym-anytrading
import gym-anytrading
import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

import numpy as np
import pandas as pd
from matlotlib import pyplot as plt

# Bring in Marketwatch GME Data
df = pd.read_csv('data/gme data.csv')

df['Date'] = pd.to_datetime(df['Date'])
 env = gym.make('stocks-v0', df=df, frame_bound=(5, 100), window_size=5)
 # env.signal_features
 state = env.reset()
 while True:
     action = env.action.sample()
	 n_state, reward, done, info = env.step(action)
	 if done:
	     print("info", info)
		 break
plt.figure(figsize=(15,6))         		 
plt.cla()
env.render_all()
plt.show()
# Build Environment and Train
env_maker = lambda: gym.make('stacks-v0', df=df. frame_bound=(5, 100), window_size=5)
env = DummyVecEnv([env_maker])

model = A2C('MlpLstmPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
# Evaluation
env = gym.make('stocks-v0', df=df, frame_bound=(90, 110), window_size=5)
obs = env.reset()
while True:
    obs = obs[np.newaxis, ...]
	action, _states = model.predict(obs)
	obs, rewards, done, info = env.step(action)
	if done:
	    print("info", info)
		break

plt.figure(figsize=(15,6))         		 
plt.cla()
env.render_all()
plt.show()		

