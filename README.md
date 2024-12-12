
## Introduction

In this work, we aim to address the dimensional collapse of MoCo v2 in medical imaging and propose two contributions to enhance the feature representations. Firstly, we introduce local feature learning that focuses on differentiating between the local regions within the input feature maps by incorporating a contrastive learning objective on the local patches of the feature maps. This helps to learn the fine-grained local features, essential for the task of medical segmentation. Secondly, we introduce feature decorrelation that uses eigenvalue decomposition to rescale and rotate the features at the final layer of the backbone. This process enhances the model’s performance by removing the correlation on the final feature map, thereby mitigating dimensional collapse during the pretraining stage

### Major features

- **Methods All in One**

  MMSelfsup provides state-of-the-art methods in self-supervised learning. For comprehensive comparison in all benchmarks, most of the pre-training methods are under the same setting.

- **Modular Design**

  MMSelfSup follows a similar code architecture of OpenMMLab projects with modular design, which is flexible and convenient for users to build their own algorithms.

- **Standardized Benchmarks**

  MMSelfSup standardizes the benchmarks including logistic regression, SVM / Low-shot SVM from linearly probed features, semi-supervised classification, object detection and semantic segmentation.

- **Compatibility**

  Since MMSelfSup adopts similar design of modulars and interfaces as those in other OpenMMLab projects, it supports smooth evaluation on downstream tasks with other OpenMMLab projects like object detection and segmentation.



## Get Started

Download Abdomen-1K and BTCV dataset (2D slices):
```
wget https://seafile.unistra.fr/d/21b27c71d0014910a823/ 
wget https://seafile.unistra.fr/d/91b8a9ec9f0246b19ddb/

```



## Installation
Having [Anaconda3](https://www.anaconda.com/products/individual#linux) installed follow the following steps for installation.
```
conda create -n med_selfsupervised python=3.8 && conda activate med_selfsupervised
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -U openmim
git clone https://github.com/open-mmlab/mmselfsup.git
cd mmselfsup/
pip install -e .
mim install mmcv==2.0.0rc4
mim install mmengine==0.7.1
cd ../ && rm -rf mmselfsup
git clone https://github.com/CAMMA-public/med_selfsup
cd pretraining/
pip install -r requirements

```

## Pretrain
Make sure of the file_paths and put file_list in the right path:
```
cp Abdomen1K-2D/train_abdomen.txt configs/selfsup/med_segmentation/mocov2_resnet50_8xb32-coslr-200e_abdomen1k_split_0_local_features_layer_4/
```
```
bash tools/dist_train.sh configs/selfsup/med_segmentation/mocov2_resnet50_8xb32-coslr-200e_abdomen1k_split_0_local_features_layer_4/mocov2_resnet50_8xb32-coslr-btcv_split_0.py \
 ${num_GPUs} \

```

## Evaluation
we use mmseg for evaluation
```
cd ..
cd evaluation/

For linear decoder:
bash tools/dist_train.sh configs/med_segmentation/linear_head_abdomen531_to_btcv_50_ft_local_layer4/deeplabv3plus_r50-d8_512x1024_40k_btcv.py \
 ${num_GPUs} \

For finetuning using DeepLabv3plus decoder:
tools/dist_train.sh configs/med_segmentation/deeplabv3_plus_r50-d8_512x512_imagenet_to_btcv_split_0/deeplabv3plus_r50-d8_512x1024_40k_btcv.py \
 ${num_GPUs} \
 



```



## Citation
we used mmselfsup and mmseg for this project. Please cite them if you use them. 
@misc{mmselfsup2021,
    title={{MMSelfSup}: OpenMMLab Self-Supervised Learning Toolbox and Benchmark},
    author={MMSelfSup Contributors},
    howpublished={\url{https://github.com/open-mmlab/mmselfsup}},
    year={2021}
}

