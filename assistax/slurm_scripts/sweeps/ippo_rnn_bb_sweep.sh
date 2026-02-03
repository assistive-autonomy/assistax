#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
cd /pvc/assistax 
ulimit -n 10000
uv run python ippo_sweep_old.py -cn ippo_sweep -m network=rnn_nps ++ENV_NAME=bedbathing ++TOTAL_TIMESTEPS=4e7 ++BATCH_SIZE=128,256,512 ++UPDATE_EPOCHS=4,8,16 "++SEED=range(0,12)" SWEEP.num_configs=12
