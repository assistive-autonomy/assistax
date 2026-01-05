#!/bin/bash

cd /home/s2618563/assistax
# --- CONFIGURATION ---
FIXED_SEEDS=12
START_CONFIGS=64   # Start high (e.g., 64 or 128)
MIN_CONFIGS=1      # Floor
DECREMENT=4        # How many configs to drop per failure
PYTHON_SCRIPT="assistax/baselines/IPPO/ippo_sweep_old.py"

echo "Starting Pareto Frontier search..."
echo "Targeting $FIXED_SEEDS seeds. Adjusting num_configs..."

current_configs=$START_CONFIGS

while [ $current_configs -ge $MIN_CONFIGS ]; do
    echo "----------------------------------------------------"
    echo "TESTING: Configs=$current_configs | Total Parallel=$((current_configs * FIXED_SEEDS))"
    echo "----------------------------------------------------"

    # Run the sweep with DISABLE_JIT=False for true memory testing
    # We use a very small TOTAL_TIMESTEPS so we only test the ALLOCATION phase
    uv run python $PYTHON_SCRIPT -cn ippo_sweep -m \
        SWEEP.num_configs=$current_configs \
        NUM_SEEDS=$FIXED_SEEDS \
        TOTAL_TIMESTEPS=100000 \
        WANDB_MODE=disabled \
        +DRY_RUN=True
    
    # Check if the last command succeeded
    if [ $? -eq 0 ]; then
        echo ""
        echo "SUCCESS! Your GPU can handle $current_configs configs with $FIXED_SEEDS seeds."
        echo "Recommended Max Setting: SWEEP.num_configs=$current_configs"
        exit 0
    else
        echo "FAILED: Out of Memory or Crash. Dropping config count..."
        current_configs=$((current_configs - DECREMENT))
        
        # Give the GPU a second to clear buffers
        sleep 2
    fi
done

echo "Error: Could not find a working configuration even at minimum levels."
exit 1