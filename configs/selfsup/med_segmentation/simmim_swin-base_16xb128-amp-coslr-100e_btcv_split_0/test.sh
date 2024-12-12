#!/bin/bash
#SBATCH -N 1
#SBATCH -p pri2018gpu
#SBATCH -c 1
#SBATCH -J Sim2_1
#SBATCH -A qoscammagpu
#SBATCH --gres=gpu:1
#SBATCH --time=180:00:00
#SBATCH --output=configs/selfsup/med_segmentation/simmim_swin-base_16xb128-amp-coslr-100e_total_segmentator_split_0/output.log
#SBATCH --export=ALL



module load slurm/slurm

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate openmmlab


sh tools/dist_train.sh configs/selfsup/med_segmentation/simmim_swin-base_16xb128-amp-coslr-100e_total_segmentator_split_0/simmim_swin-base_16xb128-amp-coslr-100e_total_segmentator.py 1 --work-dir runs/simmim_swin-base_16xb128-amp-coslr-100e_total_segmentator_split_0/