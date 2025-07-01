from dataclasses import dataclass, field

from .base import BasicGaussianExplorationConfig, BasicGaussianExploration
from .hyperplane import HyperPlaneExploration, HyperPlaneExplorationConfig
from .learned_cov import LearnedCovExplorationConfig, LearnedCovExploration


@dataclass
class StateExplorationModuleCfg:

    explore_type : str = "BasicGaussian"

    #key: "BasicGaussian"
    basic_gaussian_exploration_cfg : BasicGaussianExplorationConfig = field(default_factory=lambda: BasicGaussianExplorationConfig())

    #key: "Hyperplane"
    hyperplane_exploration_cfg : HyperPlaneExplorationConfig = field(default_factory=lambda: HyperPlaneExplorationConfig())

    #key: "LearnedCov"
    learned_cov_cfg : LearnedCovExplorationConfig = field(default_factory=lambda: LearnedCovExplorationConfig())

def make_state_exploration_module(cfg : StateExplorationModuleCfg, action_dim):

    if cfg.explore_type == "BasicGaussian":
        return BasicGaussianExploration(cfg.basic_gaussian_exploration_cfg, action_dim)
    elif cfg.explore_type == "Hyperplane":
        return HyperPlaneExploration(cfg.hyperplane_exploration_cfg, action_dim)
    elif cfg.explore_type == "LearnedCov":
        return LearnedCovExploration(cfg.learned_cov_cfg, action_dim)
