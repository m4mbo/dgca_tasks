#!/bin/bash
<<<<<<< HEAD
#SBATCH --job-name=run_%A_%a  # Job name
#SBATCH --output=logs/slurm/run_%A_%a.out
#SBATCH --error=logs/slurm/run_%A_%a.err   
#SBATCH --array=0-149%30  # 150 runs, max 30 at a time
=======
#SBATCH --job-name=dgca_run_%A_%a  # Job name
#SBATCH --output=logs/slurm/run_%A_%a.out
#SBATCH --error=logs/slurm/run_%A_%a.err   
#SBATCH --array=0-149%10  # 150 runs, max 10 at a time
>>>>>>> 56b34efcc647bef903e7e93956451c0de62f8fc3
#SBATCH --ntasks=1
#SBATCH --constraint=avx2
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
<<<<<<< HEAD
#SBATCH --time=48:00:00
=======
#SBATCH --time=24:00:00
>>>>>>> 56b34efcc647bef903e7e93956451c0de62f8fc3
#SBATCH --mem-per-cpu=2G

source /opt/modules/i12g/anaconda/3-5.0.1/bin/activate
conda activate dgca_tasks

python3 ../../run.py --run_id $SLURM_ARRAY_TASK_ID