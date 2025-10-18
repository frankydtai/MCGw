
cd /data/scratch/projects/punim0477/yitai/MaskCycleGAN-VC

module load Anaconda3/2024.02-1
conda env list
source /data/scratch/projects/punim0477/yitai/miniforge3/etc/profile.d/conda.sh

conda activate /data/scratch/projects/punim0477/yitai/miniforge3/envs/MCG

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

conda install pytorch=1.12.1 torchvision=0.13.1 torchaudio=0.12.1 cudatoolkit=10.2 -c pytorch

git add .gitignore
git commit -m "update gitignore rules"
git add -A
git commit -m "add A.py and B.py"
git push origin main

spartan.hpc.unimelb.edu.au
spartan-weather

ffmpeg -i results/TaiHF/4000hfzh/0-converted_Xinm_to_Oldm.wav \
-af "asetrate=22050*0.890898718,aresample=22050,atempo=1.122462" \
-c:a pcm_f32le -ar 22050 results/TaiHF/4000hfzh/X0-down2semis.wav

