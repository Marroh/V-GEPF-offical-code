#!/bin/bash

conda activate football
cd /home/trl/football || exit
export SDL_AUDIODRIVER=dummy

# Array of Python scripts to run
scripts=(
"light_malib/main_pbt.py --config expr_configs/cooperative_MARL_benchmark/full_game/11_vs_11_hard/mappo.yaml --wandb True --project_name guidedGRF --label random_mappo_hard_0815 --seed 1256"
"light_malib/main_pbt.py --config expr_configs/cooperative_MARL_benchmark/full_game/11_vs_11_hard/mappo.yaml --wandb True --project_name guidedGRF --label random_mappo_hard_0815 --seed 1257"
"light_malib/main_pbt.py --config expr_configs/cooperative_MARL_benchmark/full_game/11_vs_11_easy/mappo.yaml --wandb True --project_name guidedGRF --label random_mappo_easy_0815 --seed 1256"
"light_malib/main_pbt.py --config expr_configs/cooperative_MARL_benchmark/full_game/11_vs_11_easy/mappo.yaml --wandb True --project_name guidedGRF --label random_mappo_easy_0815 --seed 1257"
"light_malib/main_pbt.py --config expr_configs/cooperative_MARL_benchmark/full_game/11_vs_11_hard/mappo.yaml --wandb True --project_name guidedGRF --label random_mappo_hard_0815 --seed 1258"
"light_malib/main_pbt.py --config expr_configs/cooperative_MARL_benchmark/full_game/11_vs_11_easy/mappo.yaml --wandb True --project_name guidedGRF --label random_mappo_easy_0815 --seed 1258"
)

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

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
