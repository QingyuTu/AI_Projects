
from DQNAgent import DQNAgent
from hparams import HParams
import gymnasium as gym
import os

class TrainAgent:
    def __init__(self, model_path="new_dqnModel_lunar_lander.pt", hparams=None):
        self.model_path = model_path
        self.hparams = hparams or HParams()

    def train_agent(self):
        env = gym.make("LunarLander-v3")
        agent = DQNAgent(env=env, hparams=self.hparams, model_class=self.model_path)
        agent.save(self.model_path)
