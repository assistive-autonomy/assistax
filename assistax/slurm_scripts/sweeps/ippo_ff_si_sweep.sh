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

export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
cd /home/s2618563/assistax 
ulimit -n 10000
uv run python assistax/baselines/ippo/ippo_sweep_old.py -cn ippo_sweep -m network=ff_nps ++num_seeds=6 ++env_name=scratchitch ++total_timesteps=4e7 ++batch_size=128,256,512 ++update_epochs=4,8,16 "++seed=range(0,5)" sweep.num_configs=8
