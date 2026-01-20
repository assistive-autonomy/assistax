#!/bin/bash
#SBATCH --job-name=ippo_ff_sweep_si
#
#SBATCH --partition=PGR-Standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=5-00:00:00
#SBATCH --mem-per-cpu=16G
#
#SBATCH --gres=gpu:h200
#
#SBATCH --mail-user=l.hinckeldey@ed.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL

export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
cd /home/s2618563/assistax 
ulimit -n 10000

uv run python assistax/baselines/IPPO/ippo_sweep.py -cn ippo_sweep -m network=ff_nps ++NUM_SEEDS=6 ++ENV_NAME=scratchitch ++TOTAL_TIMESTEPS=4e7 ++NUM_MINIBATCHES=4,8,16 ++UPDATE_EPOCHS=4,8,16 "++SEED=range(0,2)" SWEEP.num_configs=8
