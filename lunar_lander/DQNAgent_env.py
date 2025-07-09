from hparams import HParams
import gymnasium as gym
import torch
import os
from utils import get_model_save_dir
from dqnModel import dqnModel

# Simplified constructor logic for demonstration
class DQNAgent:
    def __init__(self, env, model_class=None, hparams=None):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_class or os.path.join(get_model_save_dir(), "new_dqnModel_lunar_lander.pt")
        self.policy_net = dqnModel(8, 4, hparams.get_n_neurons(), hparams.get_n_layers()).to(self.device)

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
