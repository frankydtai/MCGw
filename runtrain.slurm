#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Tue Sep 02 2025 21:34:35 GMT+0800 (China Standard Time)

# Partition for the job:
#SBATCH --partition=gpu-l40s

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="MCG"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1
# The amount of memory in megabytes per node:
#SBATCH --mem=102400

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=0-9:0:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The job command(s):
#module load Anaconda3/2024.02-1
#source /data/scratch/projects/punim0477/yitai/miniforge3/etc/profile.d/conda.sh
#conda activate /data/scratch/projects/punim0477/yitai/miniforge3/envs/MaskCycleGAN-VC
#which python
#python -c "import torch; print('torch version:', torch.__version__)"

export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

python -W ignore::UserWarning -m mask_cyclegan_vc.train \
  --name TaiHF \
  --seed 0 \
  --save_dir results/ \
  --preprocessed_data_dir vcc2018_preprocessed/vcc2018_training \
  --speaker_A_id OldHF \
  --speaker_B_id NewHF \
  --epochs_per_save 1000 \
  --epochs_per_plot 10000 \
  --num_epochs 4000 \
  --batch_size 8 \
  --generator_lr 0.0002 \
  --discriminator_lr 0.0001 \
  --decay_after 2e5 \
  --sample_rate 22050 \
  --num_frames 64 \
  --max_mask_len 25 \
  --gpu_ids 0 \
  #--continue_train \
  #--start_epoch 2001 \


##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s