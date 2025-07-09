#!/bin/bash

conda activate football
cd /home/trl/football || exit

# Array of Python scripts to run  /home/trl/football/expr_configs/cooperative_MARL_benchmark/academy/counterattack/mappo.yaml
scripts=(
"light_malib/main_pbt.py --config expr_configs/cooperative_MARL_benchmark/academy/counterattack/mappo.yaml --wandb True --project_name guidedGRF_counterattack --label vgpf_0819 --seed 1258"
"light_malib/main_pbt.py --config expr_configs/cooperative_MARL_benchmark/academy/counterattack_easy/mappo.yaml --wandb True --project_name guidedGRF --label random_mappo_hard_0815 --seed 1256"
"light_malib/main_pbt.py --config expr_configs/cooperative_MARL_benchmark/academy/counterattack/mappo.yaml --wandb True --project_name guidedGRF_counterattack --label vgpf_0819 --seed 1259"
"light_malib/main_pbt.py --config expr_configs/cooperative_MARL_benchmark/academy/counterattack_easy/mappo.yaml --wandb True --project_name guidedGRF --label random_mappo_hard_0815 --seed 1257"
)

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
export CUDA_VISIBLE_DEVICES=0,1,2,5

# Loop through each script and execute it
for script in "${scripts[@]}"; do
    echo "Running $script..."
    python $script  # Use python or python3 depending on your setup

    # # Check if the script ran successfully
    # if [ $? -ne 0 ]; then
    #     echo "Error occurred while running $script. Exiting."
    #     exit 1  # Exit the script if any experiment fails
    # fi

    echo "$script completed successfully."
    echo "-----------------------------------"
done

echo "All experiments completed."
