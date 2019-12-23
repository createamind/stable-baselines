import numpy as np
import gym

env = gym.make("CarRacing-v0")
env.reset()
for _ in range(500):
    s, a, r, _ = env.step(env.action_space.sample())
    obs = env.render(mode='rgb_array')
    print(obs)