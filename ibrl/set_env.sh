# add root to python path
export PYTHONPATH=$PWD:$PYTHONPATH
#source activate ibrl

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=<YOUR MUJOCO210 PATH>
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<YOUR BIN>

# make multi-process eval work
export OMP_NUM_THREADS=1
