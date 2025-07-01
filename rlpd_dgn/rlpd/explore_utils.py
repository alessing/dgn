from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.training import train_state
from flax.training.train_state import TrainState
from typing import Any, Dict, Tuple
from functools import partial
import wandb
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions




def make_state_exploration_module(seed, layer_norm, cov_train_epochs, cov_size, update_freq, state_dim, action_dim):
    return LearnedCovExploration(LearnedCovExplorationConfig(), seed, layer_norm, cov_train_epochs, cov_size, update_freq, state_dim, action_dim)



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
    dropout : float = 0.5
    hidden_size : int = 256
    n_layers : int = 2
    lr : float = 3e-4
    batch_size : int = 128
    re_init_mlp_freq : int = 0
    weight_decay : float = 3e-2
    pert_stat_clip : float = 0.3
    cov_matrix_norm_is_max : bool = True
    entropy_coef : float = -1.
    cov_diagonal_eps : float = 1e-5

class MLP(nn.Module):
    output_shape: int
    hidden_size: int
    n_layers: int
    layer_norm: int = 0
    dropout: float = 0.0
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        for _ in range(self.n_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = nn.relu(x)
            if self.layer_norm > 0:
                x = nn.LayerNorm()(x) #TODO
            if self.dropout > 0.0:
                x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        x = nn.Dense(self.output_shape)(x)
        return x

class CovMLP(nn.Module):
    action_dim: int
    hidden_size: int
    n_layers: int
    dropout: float = 0.0
    cov_diagonal_eps: float = 1e-5
    layer_norm: int = 0
    
    @nn.compact
    def __call__(self, obs, training: bool = True):
        B = obs.shape[0]
        out_shape = self.action_dim * (self.action_dim + 1) // 2
        
        # Forward through MLP
        A_raw = MLP(
            output_shape=out_shape,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            dropout=self.dropout, 
            layer_norm=self.layer_norm
        )(obs, training=training)


        A_raw = jnp.nan_to_num(A_raw, nan=0.0, posinf=10.0, neginf=-10.0) #TODO: pen


        # A_raw = jnp.clip(A_raw, -10.0, 10.0)

        
        # Construct lower triangular matrix
        tril_indices = jnp.tril_indices(self.action_dim)
        A = jnp.zeros((B, self.action_dim, self.action_dim))
        A = A.at[:, tril_indices[0], tril_indices[1]].set(A_raw)
        
        # Make diagonal positive
        idx = jnp.arange(self.action_dim)
        A = A.at[:, idx, idx].set(nn.softplus(A[:, idx, idx]) + self.cov_diagonal_eps)

        # diag = A[:, idx, idx]
        # diag = nn.softplus(diag) + self.cov_diagonal_eps
        # diag = jnp.clip(diag, a_min=1e-4, a_max=1e3)  # clamp to avoid large/small values
        # A = A.at[:, idx, idx].set(diag)

        
        # Create distribution
        loc = jnp.zeros((B, self.action_dim))
        dist = tfd.MultivariateNormalTriL(loc=loc, scale_tril=A)
        
        return dist


@partial(jax.jit, static_argnames=("apply_fn"))
def _forward(rng, apply_fn, obs, params):    

    dropout_key, rng = jax.random.split(rng)
    key, rng = jax.random.split(rng)
    dist = apply_fn(params, obs, training=False, rngs={'dropout': dropout_key})
    
    return dist.sample(seed=key), rng


class LearnedCovExploration:
    def __init__(self, cfg: LearnedCovExplorationConfig, seed, layer_norm: int, cov_train_epochs: int, cov_size: int, update_freq: int, state_dim: int, action_dim: int):
        self.cfg = cfg
        self.layer_norm = layer_norm
        self.cov_train_epochs = cov_train_epochs
        self.cov_size = cov_size
        self.update_freq = update_freq
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.step = 0
        self.update_step = 0
        self.key = jax.random.PRNGKey(seed)
        
        # Initialize model and optimizer
        self._init_learned_cov()
        
    def _init_learned_cov(self):
        self.cov_mlp_def = CovMLP(
            action_dim=self.action_dim,
            hidden_size=self.cov_size,
            n_layers=self.cfg.n_layers,
            dropout=self.cfg.dropout,
            cov_diagonal_eps=self.cfg.cov_diagonal_eps, 
            layer_norm=self.layer_norm
        )

        
        # Initialize parameters
        self.key, init_key = jax.random.split(self.key)
        self.key, dropout_key = jax.random.split(self.key)
        dummy_obs = jnp.ones((1, self.state_dim))  # Adjust input dim as needed
        self.params = self.cov_mlp_def.init({'params': init_key, 'dropout': dropout_key}, dummy_obs)
        
        # Initialize optimizer
        if self.cfg.weight_decay > 0.:
            self.optimizer = optax.adamw(learning_rate=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        else:
            self.optimizer = optax.adam(learning_rate=self.cfg.lr)

        self.cov_mlp = TrainState.create(
            apply_fn=self.cov_mlp_def.apply,
            params=self.params,
            tx=self.optimizer,
        )
            
        
        # self.opt_state = self.optimizer.init(self.params)
    # @partial(jax.jit, static_argnames="apply_fn")
    # def _eval_actions(apply_fn, params, observations: np.ndarray) -> np.ndarray:
    #     dist = apply_fn({"params": params}, observations)
    #     return dist.mode()

    # @partial(jax.jit, static_argnames=("training", "rng"))
    # def _forward(self, obs, training=True, rngs=None):
    #     if training and rngs is None:
    #         # Generate dropout key if training and no RNGs provided
    #         self.key, dropout_key = jax.random.split(self.key)
    #         rngs = {'dropout': dropout_key}
        
    #     return self.cov_mlp.apply_fn(self.params, obs, training=training, rngs=rngs)
    
    def get_pert(self, obs):
        # self.key, sample_key = jax.random.split(self.key)
        
        # Forward pass without training mode
        pert, self.key = _forward(self.key, self.cov_mlp.apply_fn, obs, self.cov_mlp.params,)
        
        # Sample from distribution
        # pert = dist.sample(seed=sample_key)
        return pert
    
    @staticmethod
    @partial(jax.jit, static_argnames=("entropy_coef"))
    def _train_step(cov_mlp, states, delta_actions, dropout_key, entropy_coef: float):
        def loss_fn(params):
            dist = cov_mlp.apply_fn(params, states, training=True, rngs={'dropout': dropout_key})
            nll = -dist.log_prob(delta_actions).mean()
            entropy = dist.entropy().mean()
            
            loss = nll
            if entropy_coef > 0.:
                loss -= entropy_coef * entropy
            
            metrics = {
                'loss': loss,
                'nll': nll,
                'entropy': entropy
            }
            return loss, metrics
        
        # Compute loss and gradients
        grads, info = jax.grad(loss_fn, has_aux=True)(cov_mlp.params)
        cov_mlp = cov_mlp.apply_gradients(grads=grads)


        # (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        # # Update parameters
        # updates, opt_state = optimizer.update(grads, opt_state, params)
        # params = optax.apply_updates(params, updates)
        
        return cov_mlp, info
    
    def _update(self, agent, log_step):
        # Compute deltas
        states, deltas = compute_deltas_d4rl(agent, self.dataset)
        states = jnp.array(states)
        deltas = jnp.array(deltas)
        
        # Create batches
        n_samples = states.shape[0]
        indices = jnp.arange(n_samples)
        
        for _ in range(self.cov_train_epochs):


            # Generate key for shuffling
            self.key, shuffle_key = jax.random.split(self.key)
            
            # Shuffle indices using JAX
            indices = jax.random.permutation(shuffle_key, n_samples)


            # Iterate through batches
            for i in range(0, n_samples, self.cfg.batch_size):
                batch_indices = indices[i:i + self.cfg.batch_size]
                batch_states = states[batch_indices]
                batch_deltas = deltas[batch_indices]
                
                # Generate key for this step
                self.key, step_key = jax.random.split(self.key)
                
                # Update
                self.cov_mlp, metrics = self._train_step(
                    self.cov_mlp, 
                    batch_states, 
                    batch_deltas, 
                    step_key, 
                    self.cfg.entropy_coef
                )
                
                # print(f"direction mlp loss: {metrics['loss'].item()}")
        
        # Log metrics
        wandb.log({
            'Explore/loss': metrics['loss'].astype(np.float32),
            'Explore/nll': metrics['nll'].astype(np.float32),
            'Explore/entropy': metrics['entropy'].astype(np.float32),
        }, step=log_step)
        
        self.update_step += 1
    
    def update_exploration(self, agent, step=0):
        if self.step % self.update_freq == 0:
            self._update(agent, step)
        self.step += 1
    
    def set_dataset(self, dataset, dataset_type='d4rl'):
        self.dataset = dataset
        self.dataset_type = dataset_type

def compute_deltas_d4rl(agent, dataset):
    # Note: This function remains largely the same but needs adaptation
    # based on how your JAX agent works
    examples = dataset.sample(len(dataset), 
                            indx=np.arange(len(dataset)))
    
    obs = examples['observations']
    expert_actions = examples['actions']
    
    # This needs to be adapted based on your JAX agent implementation
    predicted_actions = agent.eval_actions(obs)
    action_deltas = jnp.array(expert_actions) - jnp.array(predicted_actions)
    
    obs = jnp.array(obs)
    
    return obs, action_deltas

# def make_mlp(output_shape, hidden_size, n_layers, dropout=None):
#     return MLP(
#         output_shape=output_shape,
#         hidden_size=hidden_size,
#         n_layers=n_layers,
#         dropout=dropout if dropout else 0.0
#     )
