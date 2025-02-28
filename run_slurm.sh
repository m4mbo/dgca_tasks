#!/bin/bash
#SBATCH --job-name=dgca_run_%A_%a  # Job name
#SBATCH --output=logs/run_%A_%a.out
#SBATCH --error=logs/run_%A_%a.err   
#SBATCH --array=0-149%3  # 150 runs, max 3 at a time
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=8G

conda activate dgca_tasks

python run.py --run_id $SLURM_ARRAY_TASK_ID