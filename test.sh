
cd /data/scratch/projects/punim0477/yitai/MaskCycleGAN-VC

module load Anaconda3/2024.02-1
conda env list
source /data/scratch/projects/punim0477/yitai/miniforge3/etc/profile.d/conda.sh

conda activate /data/scratch/projects/punim0477/yitai/miniforge3/envs/MaskCycleGAN-VC

sbatch runit.slurm
squeue -u $USER
scontrol show job 1
scancel -u $USER

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ffmpeg -i /vcc2018/vcc2018_evaluation/Xin/newvoice.mp3 -f segment -segment_time 3 -c pcm_s16le /vcc2018/vcc2018_evaluation/Xin/file_%03d.wav

python data_preprocessing/preprocess_vcc2018.py \
  --data_directory vcc2018/vcc2018_evaluation \
  --preprocessed_data_directory vcc2018_preprocessed/vcc2018_evaluation \
  --vocoder mel \
  --speaker_ids XinmMEL \

ffmpeg -i vcc2018/vcc2018_evaluation/Newm/dance_new.wav -t 60 vcc2018/vcc2018_evaluation/Newm/dance_new_trim.wav
