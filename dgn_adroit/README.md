
# Installation

```bash
conda create -n rlpd python=3.9 # If you use conda.
conda activate rlpd
conda install patchelf  # If you use conda.
pip install -r requirements.txt
conda deactivate
conda activate rlpd
```

# Installation

## Adroit Binary

First, download and unzip `.npy` files into `~/.datasets/awac-data/` from [here](https://drive.google.com/file/d/1yUdJnGgYit94X_AvV6JJP5Y3Lx2JF30Y/view?pli=1).


Make sure you have `mjrl` installed:
```bash
git clone https://github.com/aravindr93/mjrl
cd mjrl
pip install -e .
```

Then, recursively clone `mj_envs` from this fork:
```bash
git clone --recursive https://github.com/philipjball/mj_envs.git
```

Then sync the submodules (add the `--init` flag if you didn't recursively clone):
```bash
$ cd mj_envs  
$ git submodule update --remote
```

Finally:
```bash
$ pip install -e .
```

To test your installation, you can run:
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py --env_name=pen-binary-v0 \
                --utd_ratio=20 \
                --start_training 5000 \
                --max_steps 1000000 \
                --config=configs/rlpd_config.py \
                --config.backup_entropy=False \
                --config.hidden_dims="(256, 256, 256)" \
                --project_name=rlpd_adroit
```

# Running DGN Experiments


To run DGN, use

```
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_cov.py --env_name=door-binary-v0 \
                                --seed=${seed} \
                                --utd_ratio=20 \
                                --start_training 0 \
                                --cov_train_epochs 10 \
                                --cov_size 256 \
                                --update_freq 2000 \
                                --cov_start 50 \
                                --schedule 30000 \
                                --batch_size 128 \
                                --max_steps 1000000 \
                                --config=configs/rlpd_config.py \
                                --config.backup_entropy=False \
                                --config.hidden_dims="(256, 256, 256)" \
                                --project_name=rlpd_final_adroit
```

To run IQL, use

```
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_iql_finetuning.py --env_name=door-binary-v0 \
                                --seed=${seed} \
                                --utd_ratio=1 \
                                --start_training 0 \
                                --max_steps 1000000 \
                                --pretrain_steps 100000 \
                                --batch_size=128 \
                                --config=configs/iql_config.py \
                                --config.expectile=0.8 \
                                --config.backup_entropy=False \
                                --config.temperature=3.0 \
                                --config.hidden_dims="(256, 256, 256)" \
                                --project_name=rlpd_iql_adroit
```


# Code Credit: Reinforcement Learning with Prior Data (RLPD)


The code in this directory is a fork of the code that accompanies the paper "Efficient Online Reinforcement Learning with Offline Data", available [here](https://arxiv.org/abs/2302.02948).
