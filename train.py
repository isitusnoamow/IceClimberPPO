import retro
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import numpy as np
from gym.spaces import MultiBinary, Box
import cv2
import math

modelname = "IceClimberPPOLevel1"
gamename = "IceClimber-Nes"
#env = DummyVecEnv([lambda: retro.make(gamename, state="Level1")])


#creating env with cnn in mind
class IceClimber():
    def __init__(self):
        super().__init__()
        self.metadata = {
            'render.modes': ['human','rgb_array'],
            'video.frames_per_second': 60.0
        }   
        self.spec = None
        #220,240,3
        self.observation_space = Box(low=0,high=255,shape=(84,84,1),dtype=np.uint8)
        self.action_space = MultiBinary(9)
        self.game = retro.make(game="IceClimber-Nes",state="Level1",use_restricted_actions=retro.Actions.FILTERED)

        # for rewards
        self.height3 = True
        self.height6 = True
        self.bonus = True

    def preprocess(self,observation):
        color = cv2.cvtColor(observation,cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(color,(84,84),interpolation=cv2.INTER_CUBIC)
        channels = np.reshape(resize,(84,84,1))
        return channels
        
    def step(self,action):
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs

        if(info['height'] < self.height):
            reward = -2
        elif(info['height'] == self.height):
            reward = 0
        elif(info["height"] == 3 and self.height3):
            reward = 15
            self.height3 = False
        elif(info["height"] == 6 and self.height6):
            reward = 20
            self.height6 = False
        elif(info["height"] > 8 and self.bonus):
            reward = 100
            self.bonus = False
        else:
            reward = 1
        self.height = info['height']
        return frame_delta, reward, done, info
    
    def reset(self):
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        self.height = 0
        return obs

    def render(self, *args, **kwargs):
        self.game.render()

    def close(self):
        self.game.close()



#test if the env works 
        
# env = IceClimber()
# env = Monitor(env, './logs/')
# env = DummyVecEnv([lambda:env])
# env = VecFrameStack(env,4,channels_order='last')
# model = PPO('CnnPolicy', env, verbose=1)
# done = False
# obs = env.reset()
# while not done:
#     env.render()
#     actions, _ = model.predict(obs)
#     obs, rew, done, info = env.step(actions)
#     print(info)
# env.close()



# #tune hyperparameter
import optuna




def optimize(trial):
    return {
        'n_steps': trial.suggest_int('n_steps',2048,8192),
        'gamma': trial.suggest_loguniform('gamma',0.9,0.999), #maybe make it lower depending on how quickly ai can climb
        'learning_rate': trial.suggest_loguniform('learning_rate',1e-5,1e-4),
        'clip_range': trial.suggest_uniform('clip_range',0.1,0.3),
        'gae_lambda': trial.suggest_uniform('gae_lambda',0.9,0.999)
    }

def optimize_agent(trial):
    try:
        model_params = optimize(trial)
        env = IceClimber()
        env = Monitor(env, './logs/')
        env = DummyVecEnv([lambda:env])
        env = VecFrameStack(env,4,channels_order='last')

        model = PPO('CnnPolicy',env,verbose=0,**model_params)
        model.learn(total_timesteps=10000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        env.close()

        model.save('bestmodel')
        return mean_reward
    except Exception as e:
        print(e)
        return -1000


#implement callback somewhere here later

if __name__ == "__main__":
    experiment = optuna.create_study(direction="maximize")
    experiment.optimize(optimize_agent,n_trials=10,n_jobs=1)
    model_params = experiment.best_params
    model_params["n_steps"] = math.floor(model_params["n_steps"] / 64) * 64 #needs to be div by 64


    env = IceClimber()
    env = DummyVecEnv([lambda:env])
    env = VecFrameStack(env,4,channels_order='last')
    model = PPO('CnnPolicy', env, verbose=1, **model_params)
    model.load('bestmodel')
    model.learn(total_timesteps=100000)
    model.save('finalmodel')
