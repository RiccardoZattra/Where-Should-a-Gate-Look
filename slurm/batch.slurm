#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --time 60:00:00

#SBATCH --mem 24G
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:rtx

#SBATCH --job-name GateDependenceComparison
#SBATCH --array=0-N
#SBATCH --output outputs/batch/%A_%a_out.txt
#SBATCH --error outputs/batch/%A_%a_err.txt
#SBATCH --mail-type ALL

# 1. Define the folder containing all generated configs
CONFIG_DIR="configs/experiment/"
MODEL_DIR="configs/model/"

# 2. Read all .yaml files in the directory into bash arrays
CONFIG_FILES=(${CONFIG_DIR}/*.yaml)
MODEL_FILES=(${MODEL_DIR}/*.yaml)

NUM_CONFIGS=${#CONFIG_FILES[@]}
NUM_MODELS=${#MODEL_FILES[@]}

# 3. Map the 1D SLURM_ARRAY_TASK_ID to 2D Cartesian Product indices (Config x Model)
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_MODELS))
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_MODELS))

# 4. Select the specific files for THIS combination
CURRENT_CONFIG=${CONFIG_FILES[$CONFIG_INDEX]}
CURRENT_MODEL=${MODEL_FILES[$MODEL_INDEX]}

# 5. Extract just the filenames without the .yaml extension
CONFIG_NAME=$(basename "$CURRENT_CONFIG" .yaml)
MODEL_NAME=$(basename "$CURRENT_MODEL" .yaml)

echo "Starting task $SLURM_ARRAY_TASK_ID (Config index: $CONFIG_INDEX, Model index: $MODEL_INDEX)"
echo "Using config: $CONFIG_NAME and model: $MODEL_NAME"

# 6. Run the singularity container with the specific combination
srun singularity exec --nv singularity/afcs.sif python training_script.py experiment=$CONFIG_NAME model=$MODEL_NAME