#!/bin/bash
#SBATCH -N 1
#SBATCH -p pri2018gpu
#SBATCH -c 1
#SBATCH -J Sim0
#SBATCH -A qoscammagpu
#SBATCH --gres=gpu:8
#SBATCH --time=180:00:00
#SBATCH --output=configs/selfsup/med_segmentation/mocov2_resnet50_8xb32-coslr-200e_total_segmentator_split_0/output.log
#SBATCH --export=ALL


cd ..
cd ..
module load slurm/slurm

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate openmmlab

sh tools/dist_train.sh configs/selfsup/med_segmentation/mocov2_resnet50_8xb32-coslr-200e_total_segmentator_split_0/mocov2_resnet50_8xb32-coslr-200e_total_segmentator_split_0.py 8 --work-dir runs/simmim_swin-base_16xb128-amp-coslr-100e_total_segmentator_split_0 