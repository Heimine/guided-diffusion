#!/bin/bash
#SBATCH --mem=48g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4     # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcel-delta-gpu
#SBATCH --job-name=test
#SBATCH --time=20:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest     # <- or closest
#SBATCH --mail-user=xlxiao@umich.edu
#SBATCH --mail-type="END" #See sbatch or srun man pages for more email options


module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load anaconda3_gpu/23.9.0
source /projects/bcel/xli13/envs/final_env/bin/activate

echo "job is starting on `hostname`"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
python image2latent.py $MODEL_FLAGS --model_path /projects/bcel/xli13/Diffusion/guided-diffusion/models/256x256_diffusion_uncond.pt
