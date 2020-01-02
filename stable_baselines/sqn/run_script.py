import gym

from stable_baselines.sqn.policies import FeedForwardPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SQN
from stable_baselines.logger import configure
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
import os
target = 0.3
start = 10000
ent_coef = "auto"
name = f"EXP5_1e-3_sqn_Pong_{ent_coef}_{target}_start_{start}_s0"
os.environ['OPENAI_LOGDIR'] = '../../data/debug/' + name
os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv,tensorboard'
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-9.0/lib64:/usr/local/cuda-9.0/extras/CUPTI/lib64"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
configure()
# configure()


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
# env = gym.make('LunarLander-v2')
# env = DummyVecEnv([lambda: env])
env = make_atari_env('PongNoFrameskip-v4', num_env=1, seed=0)
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)
model = SQN(CnnPolicy, env, learning_rate=1e-3, verbose=1, seed=0, ent_coef="auto", target_entropy=target,
            buffer_size=int(1e6), learning_starts=start)
# Train the agent
model.learn(total_timesteps=int(2e6))

