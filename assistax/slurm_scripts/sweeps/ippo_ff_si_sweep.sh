#!/bin/bash
#SBATCH --job-name=ippo_ff_sweep_si
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5-00:00:00
#SBATCH --mem-per-cpu=4G
#
#SBATCH --gres=gpu:h200
#SBATCH --mail-user=l.hinckeldey@ed.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

cd /home/s2618563/assistax 
ulimit -n 10000
XLA_PYTHON_CLIENT_MEM_FRACTION=.90 uv run python assistax/baselines/IPPO/ippo_sweep_old.py -cn ippo_sweep -m network=ff_nps ++NUM_SEEDS=12 ++ENV_NAME=scratchitch ++TOTAL_TIMESTEPS=4e7 ++BATCH_SIZE=128,256,512 ++UPDATE_EPOCHS=4,8,16 "++SEED=range(0,6)" SWEEP.num_configs=24
