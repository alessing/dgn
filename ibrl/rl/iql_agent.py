from typing import Optional
from dataclasses import dataclass, field
import copy
from contextlib import contextmanager

import torch
import torch.nn as nn
import common_utils
from common_utils import ibrl_utils as utils
from networks.encoder import VitEncoder, VitEncoderConfig
from networks.encoder import ResNetEncoder, ResNetEncoderConfig, DrQEncoder
from networks.encoder import ResNet96Encoder, ResNet96EncoderConfig
from rl.actor import build_fc


from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import numpy as np
from common_utils import ibrl_utils as utils

from torch import distributions as pyd


class _QNet(nn.Module):
    def __init__(self, repr_dim, prop_dim, action_dim, feature_dim, hidden_dim, orth, drop):
        super().__init__()
        self.feature_dim = feature_dim

        self.obs_proj = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.Dropout(drop),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

        self.prop_dim = prop_dim
        q_in_dim = feature_dim + action_dim
        if prop_dim > 0:
            q_in_dim += prop_dim
        self.q = nn.Sequential(
            nn.Linear(q_in_dim, hidden_dim),
            nn.Dropout(drop),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(drop),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        if orth:
            self.apply(utils.orth_weight_init)

    def forward(self, feat, prop, action):
        assert feat.dim() == 3, f"should be [batch, patch, dim], got {feat.size()}"
        feat = feat.flatten(1, 2)
        x = self.obs_proj(feat)
        if self.prop_dim > 0:
            x = torch.cat([x, action, prop], dim=-1)
        else:
            x = torch.cat([x, action], dim=-1)
        q = self.q(x).squeeze(-1)
        return q


class _VNet(nn.Module):
    def __init__(self, repr_dim, prop_dim, feature_dim, hidden_dim, orth, drop):
        super().__init__()
        self.feature_dim = feature_dim

        self.obs_proj = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.Dropout(drop),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

        self.prop_dim = prop_dim
        q_in_dim = feature_dim
        if prop_dim > 0:
            q_in_dim += prop_dim
        self.q = nn.Sequential(
            nn.Linear(q_in_dim, hidden_dim),
            nn.Dropout(drop),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(drop),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        if orth:
            self.apply(utils.orth_weight_init)

    def forward(self, feat, prop):
        assert feat.dim() == 3, f"should be [batch, patch, dim], got {feat.size()}"
        feat = feat.flatten(1, 2)
        x = self.obs_proj(feat)
        if self.prop_dim > 0:
            x = torch.cat([x, prop], dim=-1)
        else:
            x = torch.cat([x], dim=-1)
        q = self.q(x).squeeze(-1)
        return q


@dataclass
class CriticConfig:
    feature_dim: int = 128
    hidden_dim: int = 1024
    orth: int = 1
    drop: float = 0
    fuse_patch: int = 1
    norm_weight: int = 0
    spatial_emb: int = 0


class Critic(nn.Module):
    def __init__(self, repr_dim, patch_repr_dim, prop_dim, action_dim, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.spatial_emb:
            q_cons = lambda: SpatialEmbQNet(
                fuse_patch=cfg.fuse_patch,
                num_patch=repr_dim // patch_repr_dim,
                patch_dim=patch_repr_dim,
                emb_dim=cfg.spatial_emb,
                prop_dim=prop_dim,
                action_dim=action_dim,
                hidden_dim=self.cfg.hidden_dim,
                orth=self.cfg.orth,
            )
        else:
            q_cons = lambda: _QNet(
                repr_dim=repr_dim,
                prop_dim=prop_dim,
                action_dim=action_dim,
                feature_dim=self.cfg.feature_dim,
                hidden_dim=self.cfg.hidden_dim,
                orth=self.cfg.orth,
                drop=self.cfg.drop,
            )
        self.q1 = q_cons()
        self.q2 = q_cons()

    def forward(self, feat, prop, action) -> tuple[torch.Tensor, torch.Tensor]:
        # assert self.training
        q1 = self.q1(feat, prop, action)
        q2 = self.q2(feat, prop, action)
        return q1, q2


class CriticV(nn.Module):
    def __init__(self, repr_dim, patch_repr_dim, prop_dim, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.spatial_emb:
            q_cons = lambda: SpatialEmbQNet(
                fuse_patch=cfg.fuse_patch,
                num_patch=repr_dim // patch_repr_dim,
                patch_dim=patch_repr_dim,
                emb_dim=cfg.spatial_emb,
                prop_dim=prop_dim,
                action_dim=0,
                hidden_dim=self.cfg.hidden_dim,
                orth=self.cfg.orth,
            )
        else:
            q_cons = lambda: _VNet(
                repr_dim=repr_dim,
                prop_dim=prop_dim,
                feature_dim=self.cfg.feature_dim,
                hidden_dim=self.cfg.hidden_dim,
                orth=self.cfg.orth,
                drop=self.cfg.drop,
            )
        self.q1 = q_cons()
        self.q2 = q_cons()

    def forward(self, feat, prop) -> tuple[torch.Tensor, torch.Tensor]:
        # assert self.training
        q1 = self.q1(feat, prop)
        q2 = self.q2(feat, prop)
        return q1, q2


class SpatialEmbQNet(nn.Module):
    def __init__(
        self, num_patch, patch_dim, prop_dim, action_dim, fuse_patch, emb_dim, hidden_dim, orth
    ):
        super().__init__()

        if fuse_patch:
            proj_in_dim = num_patch + action_dim + prop_dim
            num_proj = patch_dim
        else:
            proj_in_dim = patch_dim + action_dim + prop_dim
            num_proj = num_patch

        self.fuse_patch = fuse_patch
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        self.input_proj = nn.Sequential(
            nn.Linear(proj_in_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(inplace=True),
        )

        self.weight = nn.Parameter(torch.zeros(1, num_proj, emb_dim))
        nn.init.normal_(self.weight)

        self.q = nn.Sequential(
            nn.Linear(emb_dim + action_dim + prop_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        if orth:
            self.q.apply(utils.orth_weight_init)

    def extra_repr(self) -> str:
        return f"weight: nn.Parameter ({self.weight.size()})"

    def forward(self, feat: torch.Tensor, prop: torch.Tensor, action: torch.Tensor):
        assert feat.size(-1) == self.patch_dim, "are you using CNN, need flatten&transpose"

        if self.fuse_patch:
            feat = feat.transpose(1, 2)

        repeated_action = action.unsqueeze(1).repeat(1, feat.size(1), 1)
        all_feats = [feat, repeated_action]
        if self.prop_dim > 0:
            repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            all_feats.append(repeated_prop)

        x = torch.cat(all_feats, dim=-1)
        y: torch.Tensor = self.input_proj(x)
        z = (self.weight * y).sum(1)

        if self.prop_dim == 0:
            z = torch.cat((z, action), dim=-1)
        else:
            z = torch.cat((z, prop, action), dim=-1)

        q = self.q(z).squeeze(-1)
        return q


class _MultiLinear(nn.Module):
    def __init__(self, in_dim, out_dim, num_net, orth=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_net = num_net
        self.orth = orth

        self.weights = nn.Parameter(torch.zeros(num_net, in_dim, out_dim))
        self.biases = nn.Parameter(torch.zeros(1, num_net, 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.num_net):
            if self.orth:
                torch.nn.init.orthogonal_(self.weights.data[i].transpose(0, 1))
            else:
                # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
                # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
                # https://github.com/pytorch/pytorch/issues/57109
                torch.nn.init.kaiming_uniform_(self.weights.data[i].transpose(0, 1), a=math.sqrt(5))

    def __repr__(self):
        return f"_MultiLinear({self.in_dim} x {self.out_dim}, {self.num_net} nets)"

    def forward(self, x: torch.Tensor):
        """
        x: [batch, in_dim] or [batch, num_net, in_dim]
        return: [batch, num_net, out_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.num_net, -1)
        # x: [batch, num_net, in_dim]
        y = torch.einsum("bni,nio->bno", x, self.weights)
        y = y + self.biases
        return y


def _build_multi_fc(
    in_dim, action_dim, hidden_dim, num_q, num_layer, layer_norm, dropout, orth, append_action
):
    dims = [in_dim + action_dim] + [hidden_dim for _ in range(num_layer)]
    layers = []
    for i in range(num_layer):
        in_dim = dims[i]
        if append_action and i > 0:
            in_dim += action_dim
        layers.append(_MultiLinear(in_dim, dims[i + 1], num_q, orth=bool(orth)))
        if layer_norm == 1:
            layers.append(nn.LayerNorm(dims[i + 1]))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

    # layers.append(nn.Linear(dims[-1], 1))
    layers.append(_MultiLinear(dims[-1], 1, num_q, orth=bool(orth)))
    return nn.Sequential(*layers)


@dataclass
class MultiFcQConfig:
    num_q: int = 5
    num_k: int = 2
    num_layer: int = 3
    hidden_dim: int = 1024
    dropout: float = 0.0
    layer_norm: int = 1
    orth: int = 0
    append_action: int = 0


class MultiFcQ(nn.Module):
    def __init__(self, obs_shape, action_dim, cfg: MultiFcQConfig):
        super().__init__()
        assert len(obs_shape) == 1
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.cfg = cfg

        self.net = _build_multi_fc(
            in_dim=obs_shape[0],
            action_dim=action_dim,
            hidden_dim=cfg.hidden_dim,
            num_q=cfg.num_q,
            num_layer=cfg.num_layer,
            layer_norm=cfg.layer_norm,
            dropout=cfg.dropout,
            orth=cfg.orth,
            append_action=cfg.append_action,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.cfg.append_action:
            x = obs
            for layer in self.net:
                if isinstance(layer, _MultiLinear):
                    if x.dim() == 3 and action.dim() == 2:
                        action = action.unsqueeze(1).repeat(1, x.size(1), 1)
                    x = torch.cat([x, action], dim=-1)
                x = layer(x)
            y = x.squeeze(-1)
        else:
            x = torch.cat([obs, action], dim=-1)
            y = self.net(x).squeeze(-1)
        return y

    def forward_k(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        y = self.forward(obs, action)
        if self.cfg.num_k == self.cfg.num_q:
            return y

        indices = np.random.choice(self.cfg.num_q, self.cfg.num_k, replace=False)
        # y: [batch, num_q]
        selected_y = y[:, indices]
        return selected_y



class MultiFcVQ(nn.Module):
    def __init__(self, obs_shape, cfg: MultiFcQConfig):
        super().__init__()
        assert len(obs_shape) == 1
        self.obs_shape = obs_shape
        self.cfg = cfg

        self.net = _build_multi_fc(
            in_dim=obs_shape[0],
            action_dim=0,
            hidden_dim=cfg.hidden_dim,
            num_q=cfg.num_q,
            num_layer=cfg.num_layer,
            layer_norm=cfg.layer_norm,
            dropout=cfg.dropout,
            orth=cfg.orth,
            append_action=False,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if False:
            x = obs
            for layer in self.net:
                if isinstance(layer, _MultiLinear):
                    if x.dim() == 3 and action.dim() == 2:
                        action = action.unsqueeze(1).repeat(1, x.size(1), 1)
                    x = torch.cat([x, action], dim=-1)
                x = layer(x)
            y = x.squeeze(-1)
        else:
            x = torch.cat([obs], dim=-1)
            y = self.net(x).squeeze(-1)
        return y

    def forward_k(self, obs: torch.Tensor) -> torch.Tensor:
        y = self.forward(obs)
        if self.cfg.num_k == self.cfg.num_q:
            return y

        indices = np.random.choice(self.cfg.num_q, self.cfg.num_k, replace=False)
        # y: [batch, num_q]
        selected_y = y[:, indices]
        return selected_y


def test_spatial_emb_q():
    x = torch.rand(8, 144, 128)
    action = torch.rand(8, 7)

    net = SpatialEmbQNet(144, 128, 0, 7, True, 1024, 1024, False)
    print(net)
    print(net(x, prop=action, action=action).size())


if __name__ == "__main__":
    # test_proj_q()
    test_spatial_emb_q()



@dataclass
class IQLFcActorConfig:
    num_layer: int = 3
    hidden_dim: int = 1024
    dropout: float = 0.0
    layer_norm: int = 0
    orth: int = 0

    log_std_min : float = -20
    log_std_max : float = 2


class IQLFcActor(nn.Module):

    @staticmethod
    def _build_backbone(in_dim, hidden_dim, num_layer, layer_norm, dropout):
        dims = [in_dim]
        dims.extend([hidden_dim for _ in range(num_layer)])

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if layer_norm == 1:
                layers.append(nn.LayerNorm(dims[i + 1]))
            if layer_norm == 2 and (i == num_layer - 1):
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def __init__(self, obs_shape, action_dim, cfg: IQLFcActorConfig):
        super().__init__()
        assert len(obs_shape) == 1
        self.cfg = cfg

        self.net = self._build_backbone(obs_shape[0], cfg.hidden_dim, cfg.num_layer, cfg.layer_norm, cfg.dropout)
        self.mean_head = nn.Sequential(nn.Linear(self.cfg.hidden_dim, action_dim), nn.Tanh())
        
        if cfg.orth:
            self.net.apply(utils.orth_weight_init)
            self.mean_head.apply(utils.orth_weight_init)

        self.log_std_head = nn.Linear(self.cfg.hidden_dim, action_dim)



    def forward(self, obs: dict[str, torch.Tensor]):
        x = self.net(obs["state"])

        mu = self.mean_head(x)

        log_std = self.log_std_head(x)
        log_std = torch.clip(log_std, min=self.cfg.log_std_min, max=self.cfg.log_std_max)
        std = torch.exp(log_std)

        scale_tril = torch.diag_embed(std)

        #TODO: check clipping batched multiply
        dist = utils.TruncatedMultivariateNormal(loc=mu, scale_tril=scale_tril)

        return dist




@dataclass
class QAgentConfig:
    device: str = "cuda"
    lr: float = 1e-4
    critic_target_tau: float = 0.01
    stddev_clip: float = 0.3
    # special
    inv_temperature: float = 1.0
    expectile: float = 0.8
    # encoder
    use_prop: int = 0
    state_critic: MultiFcQConfig = field(default_factory=lambda: MultiFcQConfig())
    state_actor: IQLFcActorConfig = field(default_factory=lambda: IQLFcActorConfig())



class QAgent(nn.Module):
    def __init__(
        self, use_state, obs_shape, prop_shape, action_dim, rl_camera: str, cfg: QAgentConfig,
        task_idx=0, multitask=0,
    ):
        super().__init__()
        self.use_state = use_state
        self.rl_camera = rl_camera
        self.cfg = cfg
        self.multitask = multitask
        self.task_idx = task_idx

        # if self.multitask:
        #     obs_shape = (obs_shape[0] + self.multitask, )


        if use_state:
            self.critic = MultiFcQ(obs_shape, action_dim, cfg.state_critic)
            self.value_critic = MultiFcVQ(obs_shape, cfg.state_critic)
            self.actor = IQLFcActor(obs_shape, action_dim, cfg.state_actor)
        else:
            raise ValueError("Only using state observations right now")




        # self.critic = MultiFcQ(obs_shape, action_dim, cfg.state_critic)
        # self.value_critic = MultiFcVQ(obs_shape, cfg.state_critic)
        # self.actor = FcActor(obs_shape, action_dim, cfg.state_actor)

        self.critic_target = copy.deepcopy(self.critic)
        self.value_critic_target = copy.deepcopy(self.value_critic)
        self.actor_target = copy.deepcopy(self.actor)

        
        self.expectile = self.cfg.expectile
        self.inv_temperature = self.cfg.inv_temperature


        print(common_utils.wrap_ruler("critic weights"))
        print(self.critic)
        common_utils.count_parameters(self.critic)

        print(common_utils.wrap_ruler("actor weights"))
        print(self.actor)
        common_utils.count_parameters(self.actor)

        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
        self.value_critic_opt = torch.optim.Adam(self.value_critic.parameters(), lr=self.cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)

        # data augmentation
        self.aug = common_utils.RandomShiftsAug(pad=4)

        # to log rl vs bc during evaluation
        self.bc_policies: list[nn.Module] = []
        self.stats: Optional[common_utils.MultiCounter] = None

        self.critic_target.train(False)
        self.value_critic_target.train(False)
        self.train(True)
        self.to(self.cfg.device)


    def add_bc_policy(self, bc_policy):
        bc_policy.train(False)
        self.bc_policies.append(bc_policy)

    def set_stats(self, stats):
        self.stats = stats

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.value_critic.train(training)

        assert not self.critic_target.training
        assert not self.value_critic_target.training


    def _maybe_unsqueeze_(self, obs):
        should_unsqueeze = False

        if self.use_state:
            if obs["state"].dim() == 1:
                should_unsqueeze = True
        else:
            if obs[self.rl_camera].dim() == 3:
                should_unsqueeze = True

        if should_unsqueeze:
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0)
        return should_unsqueeze

    def act(
        self, obs: dict[str, torch.Tensor], *, eval_mode=False, stddev=0.0, cpu=True,
    ) -> torch.Tensor:
        """This function takes tensor and returns actions in tensor"""
        assert not self.training
        assert not self.actor.training
        unsqueezed = self._maybe_unsqueeze_(obs)


        if not self.use_state:
            assert "feat" not in obs
            obs["feat"] = self._encode(obs, augment=False)
            

        dist = self.actor.forward(obs)

        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()


        if unsqueezed:
            action = action.squeeze(0)

        action = action.detach()
        if cpu:
            action = action.cpu()
        return action



    def update_q(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        discount: torch.Tensor,
        next_obs: dict[str, torch.Tensor],
    ):
        
        assert "Only using state observation now"
        obs = obs['state']
        next_obs = next_obs['state']
        
        with torch.no_grad():
            next_v = self.value_critic_target.forward_k(next_obs).min(-1)[0]
            #NOTE: OLD LINE target_q = rewards.unsqueeze(-1) + discount * masks.unsqueeze(-1) * next_v.unsqueeze(-1)
            target_q = rewards.unsqueeze(-1) + discount.unsqueeze(-1) * next_v.unsqueeze(-1)


        critic_q = self.critic(obs, actions) # Update all 5

        critic_loss = ((critic_q - target_q)**2).sum(-1).mean()
        # critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()


        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward(retain_graph=True)
        self.critic_opt.step()



        metrics = {}
        metrics["train/critic_loss"] = critic_loss.item()
        metrics["train/critic_q"] = critic_q.mean().item()


        return metrics


    def expectile_loss(self, diff):
        weight = torch.where(diff >= 0, self.expectile, (1 - self.expectile))
        return weight * (diff**2)


    def update_v(
        self,
        obs: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor],
    ):
        
        assert "Only using state observation now"
        obs = obs['state']

        with torch.no_grad():

            critic_q = self.critic.forward_k(obs, actions)
            q = critic_q.min(-1)[0].unsqueeze(-1)
            q = q.repeat(1, self.cfg.state_critic.num_q)


        v = self.value_critic(obs)

        value_loss = self.expectile_loss(q - v)
        value_loss = value_loss.mean()


        self.value_critic_opt.zero_grad(set_to_none=True)
        value_loss.backward(retain_graph=True)
        self.value_critic_opt.step()


        metrics = {}
        metrics["train/value_loss"] = value_loss.item()
        metrics["train/v"] = v.mean().item()

        return metrics
    

    def update_actor(self, 
                     obs: dict[str, torch.Tensor], 
                     next_obs: dict[str, torch.Tensor], 
                     actions: dict[str, torch.Tensor]):

        assert self.use_state, "Right now only using state based observations"
        obs = obs['state']

        with torch.no_grad():

            v = self.value_critic(obs).min(-1)[0]

            q = self.critic(obs, actions).min(-1)[0]

            exp_a = torch.exp((q - v) * self.inv_temperature)
            exp_a = torch.minimum(exp_a, torch.full(exp_a.shape, 100.0).to('cuda'))



        dist = self.actor.forward({'state': obs})
        log_probs = dist.log_prob(actions)

        actor_loss = -(exp_a * log_probs).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward(retain_graph=True) # update critic needs retain graph debug
        self.actor_opt.step()

        metrics = {}
        metrics["train/actor_loss"] = actor_loss.item()
        metrics["train/adv"] = (q-v).mean().item()

        return metrics

    def _encode(self, obs: dict[str, torch.Tensor], augment: bool) -> torch.Tensor:
        """This function encodes the observation into feature tensor."""
        data = obs[self.rl_camera].float()
        if augment:
            data = self.aug(data)
        return self.encoder.forward(data, flatten=False)

    def _build_encoders(self, obs_shape):
        if self.cfg.enc_type == "vit":
            return VitEncoder(obs_shape, self.cfg.vit).to(self.cfg.device)
        elif self.cfg.enc_type == "resnet":
            return ResNetEncoder(obs_shape, self.cfg.resnet).to(self.cfg.device)
        elif self.cfg.enc_type == "resnet96":
            return ResNet96Encoder(obs_shape, self.cfg.resnet96).to(self.cfg.device)
        elif self.cfg.enc_type == "drq":
            return DrQEncoder(obs_shape).to(self.cfg.device)
        else:
            assert False, f"Unknown encoder type {self.cfg.enc_type}."

    def train_offline(self,
            batch,
        ):

        obs = batch.obs
        next_obs = batch.next_obs
        reward: torch.Tensor = batch.reward
        actions = batch.action["action"]
        discount: torch.Tensor = batch.bootstrap
        
        assert self.use_state, "right now only works with state observations"

        metrics = {}
        metrics["data/batch_R"] = reward.mean().item()

        value_metrics = self.update_v(obs=obs,
                                    actions=actions,)
        metrics.update(value_metrics)


        actor_metrics = self.update_actor(obs=obs,
                                        next_obs=next_obs,
                                        actions=actions,)
        metrics.update(actor_metrics)

        critic_metrics = self.update_q(obs=obs,
                                       actions=actions,
                                       rewards=reward,
                                       discount=discount,
                                       next_obs=next_obs,)
        metrics.update(critic_metrics)


        utils.soft_update_params(self.critic, self.critic_target, self.cfg.critic_target_tau)
        utils.soft_update_params(self.value_critic, self.value_critic_target, self.cfg.critic_target_tau)

        return metrics

    

    def load_critic(self, model_path, robomimic=False):
        if not model_path == "":
            if not robomimic:
                state_dict = torch.load(model_path)
                x =  list(state_dict.items())[:8]
                critic_dict = {i[0][7:]:i[1] for i in x}
            else:
                state_dict = torch.load(model_path)
                x =  list(state_dict.items())[:14]
                critic_dict = {i[0][7:]:i[1] for i in x}
            self.critic.load_state_dict(critic_dict)
            self.critic_target = copy.deepcopy(self.critic)

            self.critic_target.train(False)
            self.train(True)
            self.to(self.cfg.device)

        return
