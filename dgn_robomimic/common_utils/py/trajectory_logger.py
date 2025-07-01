import numpy as np
from pathlib import Path
from typing import Optional


class TrajectoryBuffer:
    """
    Stores (obs, act, reward, terminal, step, delta_action, episode) tuples
    and can persist / reload itself via NumPy’s .npz format.
    """

    # ------------------------------------------------------------------ #
    #  Constructor
    # ------------------------------------------------------------------ #
    def __init__(self, obs_dim: int, action_dim: int, dtype=np.float32):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.dtype = dtype
        self.reset()

    # ------------------------------------------------------------------ #
    #  Episode handling API
    # ------------------------------------------------------------------ #
    def start_episode(
        self,
        observation,
        *,
        step: Optional[int] = None,
        delta_action=None,
        episode: Optional[int] = None,
    ):
        """
        Begin a new episode with its first observation only.

        Parameters
        ----------
        observation : array-like, shape (obs_dim,)
        step : int, optional
            Logical step index for this row (defaults to 0).
        episode : int, optional
            Episode index for this row.  Defaults to `last_episode + 1`
            (or 0 if the buffer is empty).
        delta_action : array-like, optional
            Value for `delta_actions`; defaults to NaNs.
        """
        if step is None:
            step = -1
        if episode is None:
            episode = -1

        self._append(
            obs=np.asarray(observation, self.dtype).reshape(1, -1),
            act=np.full((1, self.action_dim), np.nan, self.dtype),
            rew=np.full((1,), np.nan, self.dtype),
            term=np.full((1,), np.nan, np.bool_),
            step=np.asarray([step], dtype=np.int64),
            dact=self._prep_delta(delta_action),
            ep=np.asarray([episode], dtype=np.int64),
        )

    def add_transition(
        self,
        observation,
        action,
        reward,
        terminal,
        *,
        step: Optional[int] = None,
        delta_action=None,
        episode: Optional[int] = None,
    ):
        """
        Append a complete transition.

        If *step* is omitted, defaults to `steps[-1] + 1`.
        If *episode* is omitted, defaults to `episodes[-1]`.
        """
        if step is None:
            step = -1
        if episode is None:
            episode = -1

        self._append(
            obs=np.asarray(observation, self.dtype).reshape(1, -1),
            act=np.asarray(action, self.dtype).reshape(1, -1),
            rew=np.asarray([reward], self.dtype),
            term=np.asarray([terminal], np.bool_),
            step=np.asarray([step], dtype=np.int64),
            dact=self._prep_delta(delta_action),
            ep=np.asarray([episode], dtype=np.int64),
        )

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def save(self, path, *, compressed: bool = True, **np_kw) -> None:
        """Save all arrays plus metadata to *path* (.npz)."""
        path = Path(path).with_suffix(".npz")
        savez = np.savez_compressed if compressed else np.savez

        savez(
            path,
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            terminals=self.terminals,
            steps=self.steps,
            delta_actions=self.delta_actions,
            episodes=self.episodes,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            dtype=str(self.dtype),
            **np_kw,
        )


    # ------------------------------------------------------------------ #
    #  Convenience
    # ------------------------------------------------------------------ #
    def __len__(self):
        return self.observations.shape[0]


    def reset(self):
        """Empty the buffer."""
        self.observations  = np.empty((0, self.obs_dim),   self.dtype)
        self.actions       = np.empty((0, self.action_dim), self.dtype)
        self.rewards       = np.empty((0,), self.dtype)
        self.terminals     = np.empty((0,), np.bool_)
        self.steps         = np.empty((0,), np.int64)
        self.delta_actions = np.empty((0, self.action_dim), self.dtype)
        self.episodes      = np.empty((0,), np.int64)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    def _prep_delta(self, delta_action):
        if delta_action is None:
            return np.full((1, self.action_dim), np.nan, self.dtype)
        return np.asarray(delta_action, self.dtype).reshape(1, -1)

    def _append(self, *, obs, act, rew, term, step, dact, ep):
        if obs.shape[1] != self.obs_dim:
            raise ValueError(f"observation dim {obs.shape[1]} ≠ {self.obs_dim}")
        if act.shape[1] != self.action_dim or dact.shape[1] != self.action_dim:
            raise ValueError("action / delta_action dim mismatch")

        self.observations  = np.concatenate([self.observations,  obs],  axis=0)
        self.actions       = np.concatenate([self.actions,       act],  axis=0)
        self.rewards       = np.concatenate([self.rewards,       rew],  axis=0)
        self.terminals     = np.concatenate([self.terminals,     term], axis=0)
        self.steps         = np.concatenate([self.steps,         step], axis=0)
        self.delta_actions = np.concatenate([self.delta_actions, dact], axis=0)
        self.episodes      = np.concatenate([self.episodes,      ep],   axis=0)


if __name__ == "__main__":

    buf = TrajectoryBuffer(obs_dim=2, action_dim=1)

    # new episode
    buf.start_episode([0.0, 0.0], step=0)

    # transition t=1
    buf.add_transition(
        observation=[0.1, 0.2],
        action=[1.0],
        reward=0.5,
        terminal=False,
        step=1,
        delta_action=[1.0]  # could be a control difference, for instance
    )

    print("steps:", buf.steps)                # [0 1]
    print("delta_actions:", buf.delta_actions)  # [[nan] [1.]]
    buf.save("demo.npz")


