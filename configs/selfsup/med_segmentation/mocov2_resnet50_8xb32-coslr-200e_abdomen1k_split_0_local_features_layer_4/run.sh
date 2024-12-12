#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -J moco
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#BATCH --export=ALL
#SBATCH --exclude=node-1,node-3,node-4



# module load slurm/slurm

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate myclone
cd /home/hassanpour/med_selfsupervised
bash tools/dist_train.sh configs/selfsup/med_segmentation/mocov2_resnet50_8xb32-coslr-200e_abdomen1k_split_0_local_features_layer_4/mocov2_resnet50_8xb32-coslr-btcv_split_0.py \
 1 \
 29416 \
 --work-dir=runs/mocov2_resnet50_8xb32-coslr-200e_abdomen1k_split_0_local_layer_4