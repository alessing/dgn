from dataclasses import dataclass, field
import torch
# import torch.nn as nn
import flax.linen as nn


import numpy as np
from common_utils import ibrl_utils as utils
from torch.utils.data import TensorDataset, DataLoader
import wandb
from torch.distributions import MultivariateNormal




def make_state_exploration_module(action_dim):
    return LearnedCovExploration(LearnedCovExplorationConfig(), action_dim)



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

    def set_dataset(self, dataset, dataset_type=None):
        pass



@dataclass
class LearnedCovExplorationConfig:
    update_freq : int = 1000
    dropout : float = 0.5
    hidden_size : int = 128
    n_layers : int = 2
    lr : float = 3e-4
    batch_size : int = 128
    eps_per_update : int = 100
    re_init_mlp_freq : int = 0
    weight_decay : float = 3e-2
    pert_stat_clip : float = 0.3
    cov_matrix_norm_is_max : bool = True
    entropy_coef : float = -1.
    cov_diagonal_eps : float = 1e-5

class LearnedCovExploration:

    def _init_learned_cov(self):

        out_shape = self.action_dim * (self.action_dim + 1) //2

        self.cov_mlp = make_mlp(output_shape=out_shape,
                                    hidden_size=self.cfg.hidden_size,
                                    n_layers=self.cfg.n_layers,
                                    dropout=self.cfg.dropout).to("cuda")
        params = self.cov_mlp.parameters()

        if self.cfg.weight_decay > 0.:
            self.opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        else:
            self.opt = torch.optim.Adam(params, lr=self.cfg.lr)


    def _forward(self, obs, inference=False):

        B = obs.shape[0] #batch size

        A_raw = self.cov_mlp(obs)

        tril = torch.tril_indices(self.action_dim, self.action_dim)
        A = torch.zeros(B, self.action_dim, self.action_dim, device="cuda")
        A[:, tril[0], tril[1]] = A_raw

        idx = torch.arange(self.action_dim, device="cuda")
        A[:, idx, idx] = nn.functional.softplus(A[:, idx, idx]) + self.cfg.cov_diagonal_eps


        dist = MultivariateNormal(loc= torch.zeros(B, self.action_dim, device="cuda"),
                                  scale_tril=A)
        
        return dist


    def __init__(self, cfg : LearnedCovExplorationConfig, action_dim : int):
        self.cfg : LearnedCovExplorationConfig  = cfg
        self.action_dim = action_dim
        self.step = 0
        self.update_step = 0

        self._init_learned_cov()


    def get_pert(self, obs):

        if hasattr(self, "cov_mlp"):
            self.cov_mlp.eval()

        with torch.no_grad():
            dist = self._forward(obs.to('cuda'), inference=True)

        pert = dist.sample()


        return pert


    def _update(self, agent):

        states, deltas = compute_deltas_d4rl(agent, self.dataset)


        reg_dataset = TensorDataset(torch.tensor(states).to("cuda"), torch.tensor(deltas).to("cuda"))
        reg_dataloader = DataLoader(reg_dataset, batch_size=self.cfg.batch_size,
                                    shuffle=True)
        
        if hasattr(self, "cov_mlp"):
            self.cov_mlp.train()

        for _ in range(self.cfg.eps_per_update):
            for states, delta_actions in reg_dataloader:

                dists : MultivariateNormal = self._forward(states)
                nll = -dists.log_prob(delta_actions).mean()

                entropy = dists.entropy().mean()

                loss = nll

                if self.cfg.entropy_coef > 0.:
                    loss -= self.cfg.entropy_coef*entropy

                self.opt.zero_grad()

                loss.backward()
                self.opt.step()

                print("direction mlp loss", loss.item())

        if wandb.run is not None:
            wandb.log({'Explore/loss': loss.item()}) #TODO: handle case where not using wandb
            wandb.log({'Explore/nll': nll.item()}) #TODO: handle case where not using wandb
            wandb.log({'Explore/entropy': entropy.item()}) #TODO: handle case where not using wandb
        #TODO: log eigenvalues

        self.update_step += 1

    def update_exploration(self, agent,):

        if self.step % self.cfg.update_freq == 0:
            self._update(agent)
        self.step += 1
    

    def set_dataset(self, dataset, dataset_type='d4rl'):
        self.dataset = dataset
        self.dataset_type = dataset_type



def compute_deltas_d4rl(agent, dataset):

    examples = dataset.sample(len(dataset), device="cuda", 
                              indx=np.arange(len(dataset), dtype=np.int32), )
    
    obs = examples['observations']
    expert_actions = examples['actions']

    #NOTE: currently sending entire dataset through at once. Generally this should be fine unless dataset or model is large

    predicted_actions = agent.eval_actions(obs)
    action_deltas = expert_actions.cpu().numpy() - predicted_actions.cpu().numpy()

    obs = obs.cpu().numpy()

    return obs, action_deltas


def make_mlp(output_shape, hidden_size, n_layers, dropout=None):
    layers = []
    
    for _ in range(n_layers):
        layers.append(nn.Dense(hidden_size))
        layers.append(nn.relu)
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(rate=dropout))
    
    layers.append(nn.Dense(output_shape))
    
    return nn.Sequential(layers)


# def make_mlp(output_shape, hidden_size, n_layers, dropout):

#     layers = []

#     for _ in range(n_layers):
#         layers.extend([
#             nn.LazyLinear(hidden_size),
#             nn.ReLU(),
#         ])
#         if dropout and dropout > 0.0:
#             layers.append(nn.Dropout(dropout))
#         prev_size = hidden_size

#     layers.append(nn.LazyLinear(output_shape))

#     return nn.Sequential(*layers)

