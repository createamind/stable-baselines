import gym

from stable_baselines.sac.policies import FeedForwardPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC
from stable_baselines.logger import configure

configure()

# Custom MLP policy of three layers of size 128 each
class CustomSACPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                           layers=[256, 256],
                                           layer_norm=False,
                                           feature_extraction="mlp")


env = gym.make('LunarLanderContinuous-v2')
# env = gym.make('Pendulum-v0')
env = DummyVecEnv([lambda: env])

model = SAC(CustomSACPolicy, env, verbose=1, seed=2, learning_rate=0.001, tensorboard_log="./")
# Train the agent
model.learn(total_timesteps=int(2e6))
