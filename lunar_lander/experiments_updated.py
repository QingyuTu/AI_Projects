
from train_agent_updated import TrainAgent
from hparams import HParams
import gymnasium as gym
import os
from DQNAgent import DQNAgent

def train_and_test():
    hp = HParams()
    model_path = "new_dqnModel_lunar_lander.pt"
    trainer = TrainAgent(model_path=model_path, hparams=hp)
    trainer.train_agent()

    env = gym.make("LunarLander-v3", render_mode="human")
    agent = DQNAgent(env=env, model_class=model_path, hparams=hp)
