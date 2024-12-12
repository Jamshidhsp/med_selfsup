#!/bin/bash
#SBATCH -N 1
#SBATCH -p pri2018gpu
#SBATCH -c 1
#SBATCH -J chk
#SBATCH -A qoscammagpu
#SBATCH --gres=gpu:1
#SBATCH --time=180:00:00
#SBATCH --output=total.log
#SBATCH --export=ALL



module load slurm/slurm

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate openmmlab

python tools/train.py configs/selfsup/med_segmentation/simmim_swin-base_16xb128-amp-coslr-100e_total_segmentator_split_0/simmim_swin-base_16xb128-amp-coslr-100e_in1k-192.py