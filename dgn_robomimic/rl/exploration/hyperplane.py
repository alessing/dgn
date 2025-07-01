from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np
from .explore_util import compute_deltas, make_mlp
from torch.utils.data import TensorDataset, DataLoader
import wandb

@dataclass
class HyperPlaneExplorationConfig:
    stddev : float = 0.1
    update_freq : int = 1000
    dropout : float = 0.5
    hidden_size : int = 128
    n_layers : int = 2
    lr : float = 3e-4
    batch_size : int = 128
    eps_per_update : int = 30
    wrong_dir_accept_prob : float = 0.5
    unit_vec : bool = False
    re_init_mlp_freq : int = 2

class HyperPlaneExploration:

    def _init_direction_mlp(self):

        self.direction_mlp = make_mlp(output_shape=self.action_dim,
                                      hidden_size=self.cfg.hidden_size,
                                      n_layers=self.cfg.n_layers,
                                      dropout=self.cfg.dropout).to("cuda")

        self.opt = torch.optim.Adam(self.direction_mlp.parameters(), lr=self.cfg.lr)


    def __init__(self, cfg : HyperPlaneExplorationConfig, action_dim : int):
        self.cfg : HyperPlaneExplorationConfig  = cfg
        self.action_dim = action_dim
        self.step = 0
        self.update_step = 0

        self._init_direction_mlp()

    def get_pert(self, obs):

        self.direction_mlp.eval()

        direction = self.direction_mlp(obs['state'])

        while True: #loop until get good action

            proposal_delta = self.cfg.stddev*torch.randn(1, self.action_dim, device="cuda:0")

            cosine_sim = proposal_delta[0] @ direction[0]
            if cosine_sim > 0.:
                return proposal_delta
            else:
                if np.random.uniform(low=0., high=1.0) < self.cfg.wrong_dir_accept_prob:
                    return proposal_delta

    def _update(self, agent):

        if self.cfg.re_init_mlp_freq > 0:
            if self.update_step % self.cfg.re_init_mlp_freq == 0:
                del self.direction_mlp
                del self.opt
                
                self._init_direction_mlp()

        states, deltas = compute_deltas(agent, self.dataset)

        if self.cfg.unit_vec:
            deltas = deltas / np.linalg.norm(deltas, axis=1, keepdims=True)

        reg_dataset = TensorDataset(torch.tensor(states).to("cuda"), torch.tensor(deltas).to("cuda"))
        reg_dataloader = DataLoader(reg_dataset, batch_size=self.cfg.batch_size,
                                    shuffle=True)
        
        self.direction_mlp.train()

        for _ in range(self.cfg.eps_per_update):
            for states, delta_actions in reg_dataloader:

                pred_das = self.direction_mlp(states)
                loss = nn.functional.mse_loss(pred_das, delta_actions)

                self.opt.zero_grad()

                loss.backward()
                self.opt.step()

                print("direction mlp loss", loss.item())

        wandb.log({'Explore/loss': loss.item()}) #TODO: handle case where not using wandb

        self.update_step += 1

    def update_exploration(self, agent):

        if self.step % self.cfg.update_freq == 0:
            self._update(agent)
        self.step += 1
    

    def set_dataset(self, dataset):
        self.dataset = dataset