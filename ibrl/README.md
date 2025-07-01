

## Install and compile

### Install dependencies
First Install MuJoCo

Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz)

Extract the downloaded mujoco210 directory into `~/.mujoco/mujoco210`.

### Create conda env

First create a conda env with name `dgn`.
```shell
conda create --name dgn python=3.9
```

Then, source `set_env.sh` to activate `dgn` conda env. You may need to do this to set up several important paths such as `MUJOCO_PY_MUJOCO_PATH` and add current project folder to `PYTHONPATH`.
Note that if the conda env has a different name, you will need to manually modify the `set_env.sh`.

```shell
# NOTE: run this once per shell before running any script from this repo
source set_env.sh
```

Then install python dependencies
```shell
# first install pytorch with correct cuda version, in our case we use torch 2.1 with cu121
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# then install extra dependencies from requirement.txt
pip install -r requirements.txt
```
If the command above does not work for your versions.
Please check out `tools/core_packages.txt` for a list of commands to manually install relavent packages.


### Compile C++ code
We have a C++ module in the common utils that requires compilation
```shell
cd common_utils
make
```

### Troubleshooting
Later when running the training commands, if we encounter the following error
```shell
ImportError: .../libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```
Then we can force the conda to use the system c++ lib.
Use these command to symlink the system c++ lib into conda env. To find `PATH_TO_CONDA_ENV`, run `echo ${CONDA_PREFIX:-"$(dirname $(which conda))/../"}`.

```shell
ln -sf /lib/x86_64-linux-gnu/libstdc++.so.6 PATH_TO_CONDA_ENV/bin/../lib/libstdc++.so
ln -sf /lib/x86_64-linux-gnu/libstdc++.so.6 PATH_TO_CONDA_ENV/bin/../lib/libstdc++.so.6
```

## Downloading Robomimic Demos

Demos can be downloaded from [Google Drive](https://drive.google.com/file/d/1aKM3BSJUq9D_K4384nGp5mLohKoKrj8f/view?usp=sharing). These should go in `release/data`.

## Running DGN Experiments

To run DGN for the `square` task, run

```
task="square"
num_demos=50
seed=42
num_train_step=400000
eval_eps=100
eval_interval=25000

OMP_NUM_THREADS=1 python train_rl.py --config_path release/cfgs/robomimic_rl/${task}_state_rlpd.yaml --seed $seed --save_dir exps/rl/rlpd_learned_cov_hidSize128_${task}_state_${num_demos}demos_seed${seed} --num_train_step $num_train_step --num_eval_episode $eval_eps  --log_per_step $eval_interval --mp_eval 0 --preload_num_data $num_demos --q_agent.state_actor.exploration_module.explore_type LearnedCov

```

To turn off DGN and use a standard normal noise distribution (i.e. the RLPD baseline), remove `--q_agent.state_actor.exploration_module.explore_type LearnedCov`. You can also set the value of `task` to `can`, `lift`, or `tool_hang` to run those tasks.

---

To run IBRL, you can first train a BC policy with

```
seed=1
OMP_NUM_THREADS=1 python train_bc.py --config_path release/cfgs/robomimic_bc/square_state.yaml --seed $seed --save_dir exps/bc/square_${num_demos}demos_seed${seed} --dataset.num_data $num_demos

```
Then train the RL policy with IBRL using

```
OMP_NUM_THREADS=1 python train_rl.py --config_path release/cfgs/robomimic_rl/${task}_state_ibrl.yaml --seed $seed --save_dir exps/rl/ibrl_${task}_state_${num_demos}demos_seed${seed} --num_train_step $num_train_step --num_eval_episode $eval_eps --log_per_step $eval_interval --mp_eval 0 --preload_num_data $num_demos --bc_policy exps/bc/${task}_${num_demos}demos_seed1/model0.pt
```

---

To run the IQL baseline, use

```
inv_temp=3
exp=0.8

OMP_NUM_THREADS=1 python train_iql.py --config_path iql_configs/${task}.yaml --seed $seed --save_dir exps/rl/iql_invTemp${inv_temp}_exp${exp}_${task}_state_${num_demos}demos_seed${seed} --num_train_step $num_train_step --num_eval_episode $eval_eps --log_per_step $eval_interval --mp_eval 0 --preload_num_data $num_demos --q_agent.inv_temperature $inv_temp --q_agent.expectile $exp

```




## Code Credit: IBRL 


The code within this directory is a fork of the code for _Imitation Bootstrapped Reinforcement Learning (IBRL)_ and associated baeslines (RLPD, RFT) on Robomimic.

[![Paper](https://img.shields.io/badge/Paper-%20%F0%9F%93%84-blue)](https://arxiv.org/abs/2311.02198v4)
[![Website](https://img.shields.io/badge/Website-%F0%9F%8C%90-orange)](https://ibrl.hengyuanhu.com/)


```
@misc{hu2023imitation,
    title={Imitation Bootstrapped Reinforcement Learning},
    author={Hengyuan Hu and Suvir Mirchandani and Dorsa Sadigh},
    year={2023},
    eprint={2311.02198},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
