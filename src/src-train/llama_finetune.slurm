#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=llama_finetune
#SBATCH --output=llama_finetune.%j.out
#SBATCH --error=llama_finetune.%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem-per-cpu=128GB
#SBATCH --export=ALL
#SBATCH --time=3-00:00:00

# Exit immediately if a command fails
set echo
umask 0027

module load gnu10
export USE_MAMBAPY=true
source ~/llama-r/bin/activate

# Execute your training script
python /scratch/ssenthi2/llama-r/training_llama.py
