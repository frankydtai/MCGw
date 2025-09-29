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
#SBATCH --time=0-0:10:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The job command(s):
#module load Anaconda3/2024.02-1
#conda activate /home/yitai/.conda/envs/maskvc-new2
#which python
#python -c "import torch; print('torch version:', torch.__version__)"

#export PYTHONUNBUFFERED=1
#export TQDM_DISABLE=1

python -W ignore::UserWarning -m mask_cyclegan_vc.test \
    --name Tai4 \
    --save_dir results/ \
    --converted_audio_subdir "4000zh" \
    --preprocessed_data_dir vcc2018_preprocessed/vcc2018_evaluation \
    --gpu_ids 0 \
    --speaker_A_id Old \
    --speaker_B_id New \
    --ckpt_dir results/Tai4/ckpts \
    --load_epoch 4000 \
    --model_name generator_B2A \
    --vocoder hifi_zh \

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s