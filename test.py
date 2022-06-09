from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from train import IceClimber


modelname = "finalmodel"
gamename = "IceClimber-Nes"

env = IceClimber(record=True)
env = Monitor(env, './logs/')
env = DummyVecEnv([lambda:env])
env = VecFrameStack(env,4,channels_order='last')
model = PPO.load("./pretrained/Stage1")
done = False
obs = env.reset()
while not done:
    env.render()
    actions, _ = model.predict(obs)
    obs, rew, done, info = env.step(actions)
env.close()
