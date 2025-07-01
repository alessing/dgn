from dataclasses import dataclass, field
import torch
import torch.nn as nn




@dataclass
class BasicGaussianExplorationConfig:
    stddev = 0.1

class BasicGaussianExploration:

    def __init__(self, cfg, action_dim):
        self.cfg = cfg
        self.action_dim = action_dim

    def get_pert(self, obs):
        return self.cfg.stddev*torch.randn(1, self.action_dim, device="cuda:0")

    def update_exploration(self, agent):
        pass

    def set_dataset(self, dataset):
        pass

        