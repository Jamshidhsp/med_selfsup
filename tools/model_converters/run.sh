#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -J L4
#SBATCH --gres=gpu:2
#SBATCH --time=72:00:00
#BATCH --export=ALL
#SBATCH --exclude=node-1,node-3


# module load slurm/slurm

source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate test
python tools/model_converters/convert.py /home/hassanpour/med_selfsupervised/runs/mocov2_resnet50_8xb32-coslr-200e_abdomen1k_split_0_layer_4/epoch_190.pth /home/hassanpour/med_selfsupervised/runs/mocov2_resnet50_8xb32-coslr-200e_abdomen1k_split_0_layer_4/epoch_190_converted.pth