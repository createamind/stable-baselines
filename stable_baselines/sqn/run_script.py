import gym

from stable_baselines.sqn.policies import FeedForwardPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SQN
from stable_baselines.logger import configure

configure()


# Custom MLP policy of three layers of size 128 each
class CustomSQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[256, 256],
                                           layer_norm=False,
                                           feature_extraction="mlp")

# Create and wrap the environment
# env_ = gym.make('CarRacing-v0')


# env = gym.make('Pendulum-v0')
# env = gym.make('CartPole-v0')
env = gym.make('LunarLander-v2')
# env = DummyVecEnv([lambda: env])

model = SQN(CustomSQNPolicy, env, learning_rate=0.001, verbose=1, seed=2, ent_coef=0.2)
# Train the agent
model.learn(total_timesteps=int(1e5))

