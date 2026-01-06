#!/bin/bash
#SBATCH --job-name=ippo_ff_sweep_fe
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
uv run python ippo_sweep_old.py -cn ippo_sweep -m network=ff_nps ++ENV_NAME=feeding ++TOTAL_TIMESTEPS=4e7 ++BATCH_SIZE=128,256,512 ++UPDATE_EPOCHS=4,8,16 "++SEED=range(0,12)" SWEEP.num_configs=12