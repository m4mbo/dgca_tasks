#!/bin/bash
#SBATCH --job-name=run_%A_%a  # Job name
#SBATCH --output=logs/slurm/run_%A_%a.out
#SBATCH --error=logs/slurm/run_%A_%a.err   
#SBATCH --array=0-149%30  # 150 runs, max 30 at a time
#SBATCH --ntasks=1
#SBATCH --constraint=avx2
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2G

source /opt/modules/i12g/anaconda/3-5.0.1/bin/activate
conda activate dgca_tasks

python3 ../../run.py --run_id $SLURM_ARRAY_TASK_ID