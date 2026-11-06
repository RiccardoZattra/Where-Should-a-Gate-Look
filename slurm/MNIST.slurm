#!/bin/bash

#SBATCH --job-name MNIST_Download
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --time 02:00:00
#SBATCH --mem 8G
#SBATCH --cpus-per-task 2
#SBATCH --output outputs/batch/mnist_download_%j_out.txt
#SBATCH --error outputs/batch/mnist_download_%j_err.txt
#SBATCH --mail-type ALL

# Run the preprocessing script inside the Singularity container
srun singularity exec singularity/afcs.sif python preprocessing/MNIST.py