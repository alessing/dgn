from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np
from .explore_util import compute_deltas, make_mlp
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import wandb
from torch.distributions import MultivariateNormal
from .data_util import Buffer


@dataclass
class LearnedCovExplorationConfig:
    update_freq : int = 1000
    dropout : float = 0.5
    hidden_size : int = 128
    n_layers : int = 2
    lr : float = 3e-4
    batch_size : int = 128
    eps_per_update : int = 2
    re_init_mlp_freq : int = 0
    weight_decay : float = 3e-2
    random_perturbation_boost_frac : float = 0.0
    pert_stat_clip : float = 0.3
    norm_cov_matrix : float = -1. #set to negative number to turn off
    cov_matrix_norm_is_max : bool = True
    global_cov_mat : bool = False
    matrix_type : str = "cov"
    entropy_coef : float = -1.
    cov_diagonal_eps : float = 1e-5

    first_update_num_eps : int = -1 # -1 means do the same as eps_per_update

    annealing_timescale : int = -1 # -1 means off
    annealing_shift : int = 0 # 0 means off


    add_rl_rollouts_to_demos : bool = False

    include_mean : bool = False

    shared_backbone : bool = False

    separate_mean_head_eval_mode : bool = False


    #Configs for turning off learned cov at a certain point
    lc_shutoff_step : int = -1 #shutoff at fixed step

    post_lc_std : float = 0.1

    lc_shutoff_frac : float = 0.5 # success rate fraction needed to turn off lc. -1 means off
    lc_shutoff_trigger_num_eps : int = 10 # num eps used to measure success rate
    lc_stay_shutoff : bool = False


class LearnedCovExploration:

    def _init_learned_cov(self):

        if self.cfg.matrix_type == "cov":
            out_shape = self.action_dim * (self.action_dim + 1) //2
        elif self.cfg.matrix_type == 'scalar':
            out_shape = 1
        else:
            raise ValueError("Unreckognized Gaussian Matrix Type")

        if self.cfg.global_cov_mat:
            self.cov = nn.Parameter(torch.randn(out_shape, device='cuda')/(self.action_dim**0.5),)
            params = [self.cov,]
        elif self.cfg.shared_backbone:
            self.backbone = make_mlp(output_shape=out_shape,
                                      hidden_size=self.cfg.hidden_size,
                                      n_layers=self.cfg.n_layers,
                                      dropout=self.cfg.dropout,
                                      include_output_linear=False).to("cuda")
            self.cov_head = nn.Linear(self.cfg.hidden_size,
                                      out_features=out_shape).to("cuda")
            self.mu_head = nn.Linear(self.cfg.hidden_size,
                                     self.action_dim).to("cuda")
            params = list(self.backbone.parameters()) + list(self.cov_head.parameters()) + list(self.mu_head.parameters())
        else:
            self.cov_mlp = make_mlp(output_shape=out_shape,
                                      hidden_size=self.cfg.hidden_size,
                                      n_layers=self.cfg.n_layers,
                                      dropout=self.cfg.dropout).to("cuda")
            params = self.cov_mlp.parameters()

            if self.cfg.include_mean:
                self.mean_mlp = make_mlp(output_shape=self.action_dim,
                                      hidden_size=self.cfg.hidden_size,
                                      n_layers=self.cfg.n_layers,
                                      dropout=self.cfg.dropout).to("cuda")
                params = list(self.mean_mlp.parameters()) + list(params)

        if self.cfg.weight_decay > 0.:
            self.opt = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        else:
            self.opt = torch.optim.Adam(params, lr=self.cfg.lr)


    def _forward(self, obs, inference=False):

        B = obs.shape[0] #batch size

        if self.cfg.global_cov_mat:
            A_raw = self.cov.expand(B, self.cov.shape[0])
        elif self.cfg.shared_backbone:
            x = self.backbone(obs)
            A_raw = self.cov_head(x)
        else:
            A_raw = self.cov_mlp(obs)

        if self.cfg.matrix_type == "cov":
            tril = torch.tril_indices(self.action_dim, self.action_dim)
            A = torch.zeros(B, self.action_dim, self.action_dim, device="cuda")
            A[:, tril[0], tril[1]] = A_raw
        elif self.cfg.matrix_type == 'scalar':
            I = torch.eye(self.action_dim, device=A_raw.device, dtype=A_raw.dtype).unsqueeze(0)
            A = A_raw.view(B, 1, 1) * I

        idx = torch.arange(self.action_dim, device="cuda")
        A[:, idx, idx] = nn.functional.softplus(A[:, idx, idx]) + self.cfg.cov_diagonal_eps

        if self.cfg.norm_cov_matrix > 0. and inference:
            assert A.shape[0] == 1, "for inference batch size must be one"
            frob_l2_norm = (A**2).sum()
            effective_std = torch.sqrt(frob_l2_norm/self.action_dim)
            if effective_std > self.cfg.norm_cov_matrix or not self.cfg.cov_matrix_norm_is_max:
                A = self.cfg.norm_cov_matrix*A/effective_std

        if self.cfg.include_mean:
            if self.cfg.shared_backbone:
                mu = self.mu_head(x)
            else:
                mu = self.mean_mlp(obs)
        else:
            mu = torch.zeros(B, self.action_dim, device="cuda")

        dist = MultivariateNormal(loc= mu,
                                  scale_tril=A)
        
        return dist


    def __init__(self, cfg : LearnedCovExplorationConfig, action_dim : int):
        self.cfg : LearnedCovExplorationConfig  = cfg
        self.action_dim = action_dim
        self.step = 0
        self.update_step = 0

        if self.cfg.lc_shutoff_frac > 0:
            self.log_successes = True
            self.ep_successes_list = []

        self.has_been_shutoff = False

        if self.cfg.shared_backbone:
            assert self.cfg.include_mean, "Not using mean even when have shared backbone"
            assert not self.cfg.global_cov_mat, "Not possible to have shared backbone and global cov matrix"


        self._init_learned_cov()

        if self.cfg.add_rl_rollouts_to_demos:
            self.buffer = Buffer()


    def get_pert(self, obs):

        shutoff_for_step = self.cfg.lc_shutoff_step > 0 and self.step > self.cfg.lc_shutoff_step

        
        if self.cfg.lc_shutoff_frac > 0:
            sr = self.get_recent_success_rate()
            shutoff_for_sr = sr > (self.cfg.lc_shutoff_frac - 0.0001)
            if self.cfg.lc_stay_shutoff and shutoff_for_sr:
                self.has_been_shutoff = True
            if self.has_been_shutoff:
                shutoff_for_sr = True
        else:
            shutoff_for_sr = False


        if shutoff_for_step or shutoff_for_sr:
            wandb.log({'Explore/lc_off': 1.}, step=self.step) #TODO: handle case where not using wandb
            return self.cfg.post_lc_std*torch.randn(1, self.action_dim, device="cuda:0")
        wandb.log({'Explore/lc_off': 0.}, step=self.step) #TODO: handle case where not using wandb

        if hasattr(self, "cov_mlp"):
            self.cov_mlp.eval()
        if hasattr(self, "mean_mlp"):
            if self.cfg.separate_mean_head_eval_mode:
                self.mean_mlp.eval()
        if hasattr(self, "backbone"):
            assert hasattr(self, "mu_head") and hasattr(self, "cov_head") and self.cfg.shared_backbone
            self.backbone.eval()
            self.mu_head.eval()
            self.cov_head.eval()

        with torch.no_grad():
            dist = self._forward(obs['state'], inference=True)

            pert = dist.sample()

        if self.cfg.random_perturbation_boost_frac > 0.:
            pert_clip = torch.clamp(pert, -self.cfg.pert_stat_clip, self.cfg.pert_stat_clip)
            scale = self.cfg.random_perturbation_boost_frac*torch.sqrt((pert_clip ** 2).mean())
            rand_pert = torch.randn(1, self.action_dim, device="cuda:0")
            rand_pert = rand_pert*scale

            pert += rand_pert

        if self.cfg.annealing_timescale > 0:
            anneal_factor = np.exp(-max(1, self.step - self.cfg.annealing_shift)/self.cfg.annealing_timescale)
            pert = pert*anneal_factor

        return pert


    def _update(self, agent):

        if self.cfg.use_cor_noise:
            import colorednoise as cn
            self.colored_noise_series = cn.powerlaw_psd_gaussian(self.cfg.noise_color, size=(7, self.cfg.update_freq + 2))
            self.colored_noise_step = 0

        if self.cfg.re_init_mlp_freq > 0:
            if self.update_step % self.cfg.re_init_mlp_freq == 0:
                if hasattr(self, "cov_mlp"):
                    del self.cov_mlp
                if self.cfg.shared_backbone:
                    del self.backbone
                    del self.cov_head
                    del self.mu_head
                del self.opt
                
                self._init_learned_cov()

        states, deltas = compute_deltas(agent, self.dataset)
        reg_dataset = TensorDataset(torch.tensor(states).to("cuda"), torch.tensor(deltas).to("cuda"))

        if self.cfg.add_rl_rollouts_to_demos:
            if self.buffer.has_data:
                buf_states, buf_deltas = compute_deltas(agent, self.buffer)
                buf_dataset = TensorDataset(torch.tensor(buf_states).to("cuda"), torch.tensor(buf_deltas).to("cuda"))

                reg_dataset = ConcatDataset([reg_dataset, buf_dataset])

        reg_dataloader = DataLoader(reg_dataset, batch_size=self.cfg.batch_size,
                                    shuffle=True)
        
        if hasattr(self, "cov_mlp"):
            self.cov_mlp.train()
        if hasattr(self, "mean_mlp"):
            self.mean_mlp.train()
        if hasattr(self, "backbone"):
            self.backbone.train()
        if hasattr(self, "mu_head"):
            self.mu_head.train()
        if hasattr(self, "cov_head"):
            self.cov_head.train()
        

        eps_to_train = self.cfg.eps_per_update
        if self.cfg.first_update_num_eps >= 0 and self.update_step == 0:
            eps_to_train = self.cfg.first_update_num_eps


        for _ in range(eps_to_train):
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

                print("mlp loss", loss.item())

        wandb.log({'Explore/loss': loss.item()}, step=self.step) #TODO: handle case where not using wandb
        wandb.log({'Explore/nll': nll.item()}, step=self.step) #TODO: handle case where not using wandb
        wandb.log({'Explore/entropy': entropy.item()}, step=self.step) #TODO: handle case where not using wandb
        if hasattr(self, "buffer"):
            wandb.log({'Explore/buffer_num_eps': self.buffer.num_eps}, step=self.step) #TODO: handle case where not using wandb


        self.update_step += 1

    def update_exploration(self, agent):

        if self.step % self.cfg.update_freq == 0:
            self._update(agent)

        self.step += 1
    

    def set_dataset(self, dataset):
        self.dataset = dataset

    def end_episode(self, success):
        self.ep_successes_list.append(success)

    def get_recent_success_rate(self):
        if len(self.ep_successes_list) < self.cfg.lc_shutoff_trigger_num_eps:
            return 0
        res = [float(s) for s in self.ep_successes_list[-self.cfg.lc_shutoff_trigger_num_eps:]]
        res = np.mean(res)
        return res
