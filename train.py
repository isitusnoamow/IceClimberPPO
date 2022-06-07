import retro
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

modelname = "IceClimberPPOLevel1"
gamename = "IceClimber-Nes"
env = DummyVecEnv([lambda: retro.make(gamename, state="Level1")])
model = PPO("CnnPolicy",env,n_steps=128,verbose=1)
model.set_env(env)
model.learn(total_timesteps=10000)
model.save(modelname)
env.close()

