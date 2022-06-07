import retro
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO


modelname = "IceClimberPPO"
gamename = "IceClimber-Nes"

env = DummyVecEnv([lambda: retro.make(gamename, state="Level1",record="/recordings/")])
model = PPO.load(modelname)
model.set_env(env)
obs = env.reset()
done = False
reward = 0

while not done:
    env.render()
    actions, _ = model.predict(obs)
    obs, rew, done, info = env.step(actions)
    reward += rew

print(reward)