import os
import sys
from dataclasses import dataclass, field
import yaml
import copy
from typing import Optional
import pyrallis
import torch
import numpy as np

import common_utils
from common_utils import ibrl_utils as utils
from evaluate import run_eval, run_eval_mp
from env.robosuite_wrapper import PixelRobosuite
from rl.q_agent import QAgent, QAgentConfig
from rl import replay
import train_bc


@dataclass
class MainConfig(common_utils.RunConfig):
    seed: int = 1
    # env
    task_name: str = "Lift"
    episode_length: int = 200
    end_on_success: int = 1
    # render image in higher resolution for recording or using pretrained models
    image_size: int = 224
    rl_image_size: int = 96
    rl_camera: str = "robot0_eye_in_hand"
    obs_stack: int = 1
    prop_stack: int = 1
    state_stack: int = 1
    # agent
    use_state: int = 0
    q_agent: QAgentConfig = field(default_factory=lambda: QAgentConfig())
    stddev_max: float = 1.0
    stddev_min: float = 0.1
    stddev_step: int = 500000
    nstep: int = 3
    discount: float = 0.99
    replay_buffer_size: int = 500
    batch_size: int = 256
    num_critic_update: int = 1
    update_freq: int = 2
    bc_policy: str = ""
    # rl with preload data
    mix_rl_rate: float = 1  # 1: only use rl, <1, mix in some bc data
    preload_num_data: int = 0
    preload_datapath: str = ""
    freeze_bc_replay: int = 1
    # pretrain rl policy with bc and finetune
    pretrain_only: int = 1
    pretrain_num_epoch: int = 0
    pretrain_epoch_len: int = 10000
    load_pretrained_agent: str = ""
    load_policy_only: int = 1
    add_bc_loss: int = 0
    # others
    env_reward_scale: float = 1
    num_warm_up_episode: int = 50
    num_eval_episode: int = 10
    save_per_success: int = -1
    mp_eval: int = 0  # eval with multiprocess
    num_train_step: int = 200000
    log_per_step: int = 5000
    # log
    save_dir: str = "exps/rl/debug"
    use_wb: int = 0
    replay_buf_max_ep_len : int = -1 #-1 means use episode_len

    save_all_ckpts : bool = False

    save_trajectories : bool = False

    expl_module_save_freq : int = 5000

    force_rand_warm_up : bool = False

    allow_hotfix_demo_rewards : bool = False


    def __post_init__(self):

        if self.replay_buf_max_ep_len < 0:
            self.replay_buf_max_ep_len = self.episode_length

        self.rl_cameras = self.rl_camera.split("+")

        if self.bc_policy in ["none", "None"]:
            self.bc_policy = ""

        if self.bc_policy:
            print(f"Using BC policy {self.bc_policy}")
            os.path.exists(self.bc_policy)

        if self.pretrain_num_epoch > 0:
            assert self.preload_num_data > 0

        self.stddev_min = min(self.stddev_max, self.stddev_min)

        if self.preload_datapath:
            self.num_warm_up_episode += self.preload_num_data

        if self.task_name == "TwoArmTransport":
            self.robots: list[str] = ["Panda", "Panda"]
        else:
            self.robots: list[str] = ["Panda"]

    @property
    def bc_cameras(self) -> list[str]:
        if not self.bc_policy:
            return []

        bc_cfg_path = os.path.join(os.path.dirname(self.bc_policy), f"cfg.yaml")
        bc_cfg = pyrallis.load(train_bc.MainConfig, open(bc_cfg_path, "r"))  # type: ignore
        return bc_cfg.dataset.rl_cameras

    @property
    def stddev_schedule(self):
        return f"linear({self.stddev_max},{self.stddev_min},{self.stddev_step})"


class Workspace:

    def _get_dataset(self):
        cfg = self.cfg

        from bc.dataset import DatasetConfig, RobomimicDataset
        dataset_config = DatasetConfig()
        dataset_config.path = cfg.preload_datapath
        dataset_config.num_data = cfg.preload_num_data
        dataset_config.use_state = cfg.use_state
        dataset_config.state_stack = cfg.state_stack
        dataset_config.obs_stack = cfg.obs_stack
        dataset_config.prop_stack = cfg.prop_stack
        dataset_config.rl_camera = cfg.rl_camera
        dataset_config.eval_episode_len = cfg.episode_length
        dataset_config.allow_long_episodes = 1
        dataset_config.__post_init__()
        dataset = RobomimicDataset(dataset_config)

        return dataset

    def __init__(self, cfg: MainConfig, from_main=True):
        self.work_dir = cfg.save_dir
        print(f"workspace: {self.work_dir}")

        print("LOG PATH", cfg.log_path)

        if from_main:

            

            if os.path.exists(os.path.dirname(cfg.log_path)) and ("debug" not in cfg.log_path):
                raise ValueError("Already did this run. If want to overwrite, delete log folder")
                return
        
            common_utils.set_all_seeds(cfg.seed)
            sys.stdout = common_utils.Logger(cfg.log_path, print_to_stdout=True)

            pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
            print(common_utils.wrap_ruler("config"))
            with open(cfg.cfg_path, "r") as f:
                print(f.read(), end="")
            print(common_utils.wrap_ruler(""))

        self.cfg = cfg
        self.cfg_dict = yaml.safe_load(open(cfg.cfg_path, "r"))

        self.global_step = 0
        self.global_episode = 0
        self.train_step = 0
        self._setup_env()

        print(self.train_env.observation_shape)
        self.agent = QAgent(
            self.cfg.use_state,
            self.train_env.observation_shape,
            self.train_env.prop_shape,
            self.train_env.action_dim,
            self.cfg.rl_camera,
            cfg.q_agent,
        )

        if not from_main:
            return

        if cfg.load_pretrained_agent and cfg.load_pretrained_agent != "None":
            print(f"loading loading pretrained agent from {cfg.load_pretrained_agent}")
            critic_states = copy.deepcopy(self.agent.critic.state_dict())
            self.agent.load_state_dict(torch.load(cfg.load_pretrained_agent))
            if cfg.load_policy_only:
                # avoid overwriting critic
                self.agent.critic.load_state_dict(critic_states)
                self.agent.critic_target.load_state_dict(critic_states)

        self.ref_agent = copy.deepcopy(self.agent)
        # override to always use RL even when self.agent is ibrl
        self.ref_agent.cfg.act_method = "rl"

        if len(self.cfg.preload_datapath) > 0:
            self.agent.set_explore_dataset(self._get_dataset())

        # set up bc related stuff
        self.bc_policy: Optional[torch.nn.Module] = None
        if cfg.bc_policy:
            bc_policy, _, bc_env_params = train_bc.load_model(cfg.bc_policy, "cuda")
            assert bc_env_params["obs_stack"] == self.eval_env_params["obs_stack"]

            self.agent.add_bc_policy(copy.deepcopy(bc_policy))
            self.bc_policy = bc_policy

        self._setup_replay()

    def _setup_env(self):
        self.rl_cameras: list[str] = list(set(self.cfg.rl_cameras + self.cfg.bc_cameras))
        if self.cfg.use_state:
            self.rl_cameras = []
        print(f"rl_cameras: {self.rl_cameras}")

        if self.cfg.save_per_success > 0:
            for cam in ["agentview", "robot0_eye_in_hand"]:
                if cam not in self.rl_cameras:
                    print(f"Adding {cam} to recording camera because {self.cfg.save_per_success=}")
                    self.rl_cameras.append(cam)

        self.obs_stack = self.cfg.obs_stack
        self.prop_stack = self.cfg.prop_stack

        self.train_env = PixelRobosuite(
            env_name=self.cfg.task_name,
            robots=self.cfg.robots,
            episode_length=self.cfg.episode_length,
            reward_shaping=False,
            image_size=self.cfg.image_size,
            rl_image_size=self.cfg.rl_image_size,
            camera_names=self.rl_cameras,
            rl_cameras=self.rl_cameras,
            env_reward_scale=self.cfg.env_reward_scale,
            end_on_success=bool(self.cfg.end_on_success),
            use_state=bool(self.cfg.use_state),
            obs_stack=self.obs_stack,
            state_stack=self.cfg.state_stack,
            prop_stack=self.prop_stack,
            record_sim_state=bool(self.cfg.save_per_success > 0),
        )
        self.eval_env_params = dict(
            env_name=self.cfg.task_name,
            robots=self.cfg.robots,
            episode_length=self.cfg.episode_length,
            reward_shaping=False,
            image_size=self.cfg.image_size,
            rl_image_size=self.cfg.rl_image_size,
            camera_names=self.rl_cameras,
            rl_cameras=self.rl_cameras,
            use_state=self.cfg.use_state,
            obs_stack=self.obs_stack,
            state_stack=self.cfg.state_stack,
            prop_stack=self.prop_stack,
        )
        self.eval_env = PixelRobosuite(**self.eval_env_params)  # type: ignore

    def _setup_replay(self):
        use_bc = False
        if self.cfg.mix_rl_rate < 1:
            use_bc = True
        if self.cfg.save_per_success > 0:
            use_bc = True
        if self.cfg.pretrain_num_epoch > 0 or self.cfg.add_bc_loss:
            assert self.cfg.preload_num_data
            use_bc = True

        self.replay = replay.ReplayBuffer(
            self.cfg.nstep,
            self.cfg.discount,
            frame_stack=1,
            max_episode_length=self.cfg.replay_buf_max_ep_len,
            replay_size=self.cfg.replay_buffer_size,
            use_bc=use_bc,
            save_per_success=self.cfg.save_per_success,
            save_dir=self.cfg.save_dir,
        )

        if self.cfg.preload_num_data:
            replay.add_demos_to_replay(
                self.replay,
                self.cfg.preload_datapath,
                num_data=self.cfg.preload_num_data,
                rl_cameras=self.rl_cameras,
                use_state=self.cfg.use_state,
                obs_stack=self.obs_stack,
                state_stack=self.cfg.state_stack,
                prop_stack=self.prop_stack,
                reward_scale=self.cfg.env_reward_scale,
                record_sim_state=bool(self.cfg.save_per_success > 0),
                allow_hotfix_demo_rewards=self.cfg.allow_hotfix_demo_rewards
            )
        if self.cfg.freeze_bc_replay:
            assert self.cfg.save_per_success <= 0, "cannot save a non-growing replay"
            self.replay.freeze_bc_replay = True

    def eval(self, seed, policy) -> float:
        random_state = np.random.get_state()

        if self.cfg.mp_eval:
            scores: list[float] = run_eval_mp(
                env_params=self.eval_env_params,
                agent=policy,
                num_proc=10,
                num_game=self.cfg.num_eval_episode,
                seed=seed,
                verbose=False,
            )
        else:
            scores: list[float] = run_eval(
                env_params=self.eval_env_params,
                agent=policy,
                num_game=self.cfg.num_eval_episode,
                seed=seed,
                record_dir=None,
                verbose=False,
            )

        np.random.set_state(random_state)
        return float(np.mean(scores))  # type: ignore

    def warm_up(self):
        # warm up stage, fill the replay with some episodes
        # it can either be human demos, or generated by the bc, or purely random
        obs, _ = self.train_env.reset()
        self.replay.new_episode(obs)
        total_reward = 0
        num_episode = 0
        while True:
            if self.bc_policy is not None and not self.cfg.force_rand_warm_up:
                # we have a BC policy
                with torch.no_grad(), utils.eval_mode(self.bc_policy):
                    action = self.bc_policy.act(obs, eval_mode=True)
            elif self.cfg.load_pretrained_agent or self.cfg.pretrain_num_epoch > 0:
                # the policy has been pretrained/initialized
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs, eval_mode=True)
            else:
                action = torch.zeros(self.train_env.action_dim)
                action = action.uniform_(-1.0, 1.0)

            obs, reward, terminal, success, image_obs = self.train_env.step(action)
            reply = {"action": action}
            self.replay.add(obs, reply, reward, terminal, success, image_obs)

            if terminal:
                num_episode += 1
                total_reward += self.train_env.episode_reward
                if self.replay.size() < self.cfg.num_warm_up_episode:
                    self.replay.new_episode(obs)
                    obs, _ = self.train_env.reset()
                else:
                    break

        print(f"Warm up done. #episode: {self.replay.size()}")
        print(f"#episode from warmup: {num_episode}, #reward: {total_reward}")

    def train(self):

        print("Start Training Function")


        stat = common_utils.MultiCounter(
            self.work_dir,
            bool(self.cfg.use_wb),
            wb_exp_name=self.cfg.wb_exp,
            wb_run_name=self.cfg.wb_run,
            wb_group_name=self.cfg.wb_group,
            config=self.cfg_dict,
        )
        self.agent.set_stats(stat)
        saver = common_utils.TopkSaver(save_dir=self.work_dir, topk=1)

        if self.cfg.expl_module_save_freq > 0: 
            os.makedirs(os.path.join(self.work_dir, "expl_module_ckpts"), exist_ok=True)
            

        if self.replay.num_episode < self.cfg.num_warm_up_episode:
            print("Starting Warm Up")
            self.warm_up()

        print("Getting first obs in training loop")

        stopwatch = common_utils.Stopwatch()
        obs, _ = self.train_env.reset()
        self.replay.new_episode(obs)

        if self.cfg.save_trajectories:
            self.trajectory_saver = common_utils.TrajectoryBuffer(obs_dim=self.eval_env.observation_shape[0],
            action_dim=self.eval_env.action_dim,)

            self.trajectory_saver.start_episode(obs['state'].cpu().numpy(), step=self.global_step, episode=self.global_episode)

        while self.global_step < self.cfg.num_train_step:

            # Update exploration module -- internally this function decides how often to do updates
            self.agent.update_exploration()

            ### act ###
            with stopwatch.time("act"), torch.no_grad(), utils.eval_mode(self.agent):
                stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
                action = self.agent.act(obs, eval_mode=False, stddev=stddev, explore_mode=True)
                stat["data/stddev"].append(stddev)

            ## log to Exploration Module Buffer if Applicable
            if hasattr(self.agent.actor.explore_module, "buffer"):
                self.agent.actor.explore_module.buffer.add_transition(obs['state'], action)

            ### env.step ###
            with stopwatch.time("env step"):
                obs, reward, terminal, success, image_obs = self.train_env.step(action)

                if self.cfg.save_trajectories:
                    self.trajectory_saver.add_transition(observation=obs['state'].cpu().numpy(),
                    action=action, reward=reward,
                    terminal=terminal,
                    delta_action=self.agent.actor.most_recent_pert,
                    step=self.global_step,
                    episode=self.global_episode)

                if self.global_step % 100 == 0:
                    print("global_step", self.global_step)
                    wandb.log({'progress/global_step': self.global_step}) #TODO: handle case where not using wandb

            with stopwatch.time("add"):
                assert isinstance(terminal, bool)
                reply = {"action": action}
                self.replay.add(obs, reply, reward, terminal, success, image_obs)
                self.global_step += 1

            if self.cfg.expl_module_save_freq > 0:
                if self.global_step % self.cfg.expl_module_save_freq == 0:
                    if hasattr(self.agent.actor.explore_module, "cov_mlp"):
                        cov_mlp_state_dict = self.agent.actor.explore_module.cov_mlp.state_dict()

                        torch.save(cov_mlp_state_dict, os.path.join(self.work_dir, "expl_module_ckpts", f"cov_mlp_model_step{self.global_step}.pt"))

            if terminal:
                with stopwatch.time("reset"):
                    self.global_episode += 1
                    stat["score/train_score"].append(float(success))
                    stat["data/episode_len"].append(self.train_env.time_step)

                    if hasattr(self.agent.actor.explore_module, "buffer"):
                        self.agent.actor.explore_module.buffer.end_episode(success)
                    if getattr(self.agent.actor.explore_module, "log_successes", False):
                        self.agent.actor.explore_module.end_episode(success)

                    # reset env
                    obs, _ = self.train_env.reset()
                    self.replay.new_episode(obs)

                    if self.cfg.save_trajectories:
                        self.trajectory_saver.start_episode(obs['state'].cpu().numpy(), step=self.global_step, episode=self.global_episode)
                    
                    
                    

            ### logging ###
            if self.global_step % self.cfg.log_per_step == 0:
                self.log_and_save(stopwatch, stat, saver)

            ### train ###
            if self.global_step % self.cfg.update_freq == 0:
                with stopwatch.time("train"):
                    self.rl_train(stat)
                    self.train_step += 1

    def log_and_save(
        self,
        stopwatch: common_utils.Stopwatch,
        stat: common_utils.MultiCounter,
        saver: common_utils.TopkSaver,
    ):
        elapsed_time = stopwatch.elapsed_time_since_reset
        stat["other/speed"].append(self.cfg.log_per_step / elapsed_time)
        stat["other/elapsed_time"].append(elapsed_time)
        stat["other/episode"].append(self.global_episode)
        stat["other/step"].append(self.global_step)
        stat["other/train_step"].append(self.train_step)
        stat["other/replay"].append(self.replay.size())
        stat["score/num_success"].append(self.replay.num_success)

        if self.replay.bc_replay is not None:
            stat["data/bc_replay_size"].append(self.replay.size(bc=True))

        with stopwatch.time("eval"):
            eval_seed = (self.global_step // self.cfg.log_per_step) * self.cfg.num_eval_episode
            stat["eval/seed"].append(eval_seed)
            eval_score = self.eval(seed=eval_seed, policy=self.agent)
            stat["score/score"].append(eval_score)

            original_act_method = self.agent.cfg.act_method
            # if self.agent.cfg.act_method != "rl":
            #     with self.agent.override_act_method("rl"):
            #         rl_score = self.eval(seed=eval_seed, policy=self.agent)
            #         stat["score/score_rl"].append(rl_score)
            #         stat["score_diff/hybrid-rl"].append(eval_score - rl_score)

            if self.agent.cfg.act_method == "ibrl_soft":
                with self.agent.override_act_method("ibrl"):
                    greedy_score = self.eval(seed=eval_seed, policy=self.agent)
                    stat["score/greedy_score"].append(greedy_score)
                    stat["score_diff/greedy-soft"].append(greedy_score - eval_score)
            assert self.agent.cfg.act_method == original_act_method


        force_save_name = f'model_step{self.global_step}' if self.cfg.save_all_ckpts else None

        saved = saver.save(self.agent.state_dict(), eval_score, save_latest=True, force_save_name=force_save_name)
        stat.summary(self.global_step, reset=True)
        print(f"saved?: {saved}")
        stopwatch.summary(reset=True)
        print("total time:", common_utils.sec2str(stopwatch.total_time))
        print(common_utils.get_mem_usage())

        if self.cfg.save_trajectories:
            self.trajectory_saver.save(os.path.join(self.work_dir, f"trajectory{self.global_step}"))
            self.trajectory_saver.reset()
            

    def rl_train(self, stat: common_utils.MultiCounter):
        stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
        for i in range(self.cfg.num_critic_update):
            if self.cfg.mix_rl_rate < 1:
                rl_bsize = int(self.cfg.batch_size * self.cfg.mix_rl_rate)
                bc_bsize = self.cfg.batch_size - rl_bsize
                batch = self.replay.sample_rl_bc(rl_bsize, bc_bsize, "cuda:0")
            else:
                batch = self.replay.sample(self.cfg.batch_size, "cuda:0")

            # in RED-Q, only update actor once
            update_actor = i == self.cfg.num_critic_update - 1

            bc_batch = None
            if update_actor and self.cfg.add_bc_loss:
                bc_batch = self.replay.sample_bc(self.cfg.batch_size, "cuda:0")

            metrics = self.agent.update(batch, stddev, update_actor, bc_batch, self.ref_agent)

            stat.append(metrics)
            stat["data/discount"].append(batch.bootstrap.mean().item())

    def pretrain_policy(self):
        stat = common_utils.MultiCounter(
            self.work_dir,
            bool(self.cfg.use_wb),
            wb_exp_name=self.cfg.wb_exp,
            wb_run_name=self.cfg.wb_run,
            wb_group_name=self.cfg.wb_group,
            config=self.cfg_dict,
        )
        saver = common_utils.TopkSaver(save_dir=self.work_dir, topk=1)

        for epoch in range(self.cfg.pretrain_num_epoch):
            for _ in range(self.cfg.pretrain_epoch_len):
                batch = self.replay.sample_bc(self.cfg.batch_size, "cuda")
                metrics = self.agent.pretrain_actor_with_bc(batch)

                for k, v in metrics.items():
                    stat[k].append(v)

            eval_seed = epoch * self.cfg.pretrain_epoch_len
            score = self.eval(eval_seed, policy=self.agent)
            stat["pretrain/score"].append(score)

            stat.summary(epoch, reset=True)
            saved = saver.save(self.agent.state_dict(), score, save_latest=True)
            print(f"saved?: {saved}")
            print(common_utils.get_mem_usage())


def load_model(weight_file, device, load_dataset=False):
    cfg_path = os.path.join(os.path.dirname(weight_file), f"cfg.yaml")
    print(common_utils.wrap_ruler("config of loaded agent"))
    with open(cfg_path, "r") as f:
        print(f.read(), end="")
    print(common_utils.wrap_ruler(""))

    cfg = pyrallis.load(MainConfig, open(cfg_path, "r"))  # type: ignore
    cfg.save_dir = os.path.dirname(weight_file)
    
    original_preload_num_data = cfg.preload_num_data
    cfg.preload_num_data = 0  # override this to avoid loading data

    workplace = Workspace(cfg, from_main=False)

    eval_env = workplace.eval_env
    eval_env_params = workplace.eval_env_params
    agent = workplace.agent
    state_dict = torch.load(weight_file)
    agent.load_state_dict(state_dict)

    if cfg.bc_policy:
        bc_policy = train_bc._load_model(cfg.bc_policy, eval_env, device)
        agent.add_bc_policy(bc_policy)

    if original_preload_num_data > 0 and load_dataset:
        workplace.cfg.preload_num_data = original_preload_num_data
        agent.set_explore_dataset(workplace._get_dataset())

    agent = agent.to(device)
    return agent, eval_env, eval_env_params


def main():
    cfg = pyrallis.parse(config_class=MainConfig)  # type: ignore

    print("Done parsing")

    workspace = Workspace(cfg)
    if cfg.pretrain_num_epoch > 0:
        print("Pretraining")
        workspace.pretrain_policy()
        if not cfg.pretrain_only:
            print("RL finetuning")
            workspace.train()
    else:
        workspace.train()

    if cfg.use_wb:
        wandb.finish()

    assert False


if __name__ == "__main__":
    import wandb
    from rich.traceback import install

    install()
    os.environ["MUJOCO_GL"] = "egl"
    torch.backends.cudnn.allow_tf32 = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    main()
