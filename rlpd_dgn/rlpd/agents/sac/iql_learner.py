"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax
import gym
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from rlpd.agents.agent import Agent
from rlpd.agents.sac.temperature import Temperature
from rlpd.data.dataset import DatasetDict
from rlpd.distributions import TanhNormal, IQLTanhNormal
from rlpd.networks import (
    MLP,
    Ensemble,
    MLPResNetV2,
    StateActionValue,
    StateValue, 
    subsample_ensemble,
)


# From https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb#scrollTo=ap-zaOyKJDXM
def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))



def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)



class IQL(Agent):
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    temp: TrainState
    tau: float
    temperature: float
    discount: float
    expectile: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    backup_entropy: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        expectile: float = 0.8, 
        temperature: float = 1.0, 
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        use_pnorm: bool = False,
        use_critic_resnet: bool = False,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key, value_key = jax.random.split(rng, 5)

        actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, activate_final=True, use_layer_norm=False, use_pnorm=use_pnorm
        )
        actor_def = IQLTanhNormal(actor_base_cls, action_dim, log_std_scale=1e-3, log_std_min=-5.0)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        if use_critic_resnet:
            critic_base_cls = partial(
                MLPResNetV2,
                num_blocks=1,
            )
        else:
            critic_base_cls = partial(
                MLP,
                hidden_dims=hidden_dims,
                activate_final=True,
                dropout_rate=critic_dropout_rate,
                use_layer_norm=critic_layer_norm,
                use_pnorm=use_pnorm,
            )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=tx,
        )


        if use_critic_resnet:
            value_base_cls = partial(
                MLPResNetV2,
                num_blocks=1,
            )
        else:
            value_base_cls = partial(
                MLP,
                hidden_dims=hidden_dims,
                activate_final=True,
                dropout_rate=critic_dropout_rate,
                use_layer_norm=critic_layer_norm,
                use_pnorm=use_pnorm,
            )
        value_def = StateValue(base_cls=value_base_cls)
        value_params = value_def.init(value_key, observations)["params"]
        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=critic_lr)
        value = TrainState.create(
            apply_fn=value_def.apply,
            params=value_params,
            tx=tx,
        )


        
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            value=value, 
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            expectile=expectile,
            discount=discount,
            temperature=temperature,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
        )

    

    def update_value(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:

        key, rng = jax.random.split(self.rng)
        qs = self.target_critic.apply_fn(
            {"params": self.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        q = qs.min(axis=0)

        def value_loss_fn(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = self.value.apply_fn({"params": value_params}, batch["observations"])
            value_loss = expectile_loss(q - v, self.expectile).mean()

            return value_loss, {"value_loss": value_loss, "v": v.mean()}

        grads, info = jax.grad(value_loss_fn, has_aux=True)(self.value.params)
        value = self.value.apply_gradients(grads=grads)
        return self.replace(value=value, rng=rng), info


    def update_actor(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)


        v = self.value.apply_fn(
            {"params": self.value.params},
            batch["observations"],
            False,
            rngs={"dropout": key},
        )  # training=True
        

        qs = self.critic.apply_fn(
            {"params": self.critic.params},
            batch["observations"],
            batch["actions"],
            False,
            rngs={"dropout": key},
        )  # training=True
        q = qs.min(axis=0)



        exp_a = jnp.exp((q - v) * self.temperature)
        # exp_a = jnp.exp(temperature)
        exp_a = jnp.minimum(exp_a, 100.0)


        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])

            batch_actions = batch["actions"]

            log_probs = dist.log_prob(batch_actions)


            # log_probs = jnp.clip(log_probs, -50, 0)


            actor_loss = - (
                exp_a * log_probs
            ).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)


        grads, _ = optax.clip_by_global_norm(1.0).update(grads, state=None)


        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)

        grads, _ = optax.clip_by_global_norm(1.0).update(grads, state=None)


        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:

        dist = self.actor.apply_fn(
            {"params": self.actor.params}, batch["next_observations"]
        )

        rng = self.rng

        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)

        key, rng = jax.random.split(rng)
        next_v = self.value.apply_fn(
            {"params": self.value.params},
            batch["next_observations"],
            False,
            rngs={"dropout": key},
        )  # training=True

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_v

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)


        grads, _ = optax.clip_by_global_norm(1.0).update(grads, state=None)


        critic = self.critic.apply_gradients(grads=grads)


        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info


    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)


        new_agent, value_info = new_agent.update_value(mini_batch)
        new_agent, actor_info = new_agent.update_actor(mini_batch)

        return new_agent, {**actor_info, **critic_info}


