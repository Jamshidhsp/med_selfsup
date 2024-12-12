<div align="center">
  <img src="./resources/mmselfsup_logo.png" width="500"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/mmselfsup)](https://pypi.org/project/mmselfsup)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmselfsup.readthedocs.io/en/dev-1.x/)
[![badge](https://github.com/open-mmlab/mmselfsup/workflows/build/badge.svg)](https://github.com/open-mmlab/mmselfsup/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmselfsup/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmselfsup)
[![license](https://img.shields.io/github/license/open-mmlab/mmselfsup.svg)](https://github.com/open-mmlab/mmselfsup/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmselfsup.svg)](https://github.com/open-mmlab/mmselfsup/issues)

[📘Documentation](https://mmselfsup.readthedocs.io/en/dev-1.x/) |
[🛠️Installation](https://mmselfsup.readthedocs.io/en/dev-1.x/get_started.html) |
[👀Model Zoo](https://mmselfsup.readthedocs.io/en/dev-1.x/model_zoo.html) |
[🆕Update News](https://mmselfsup.readthedocs.io/en/dev-1.x/notes/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmselfsup/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

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

## Installation

MMSelfSup depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv), [MMEngine](https://github.com/open-mmlab/mmengine) and [MMClassification](https://github.com/open-mmlab/mmclassification).

Please refer to [Installation](https://mmselfsup.readthedocs.io/en/dev-1.x/get_started.html) for more detailed instruction.

## Get Started

Pretrain
Download Abdomen-1K dataset (2D slices).
Downetream Tasks
Download BTCV dataset (2D slices)

- [Classification](https://mmselfsup.readthedocs.io/en/dev-1.x/user_guides/classification.html)
- [Detection](https://mmselfsup.readthedocs.io/en/dev-1.x/user_guides/detection.html)
- [Segmentation](https://mmselfsup.readthedocs.io/en/dev-1.x/user_guides/segmentation.html)

Useful Tools

- [Visualization](https://mmselfsup.readthedocs.io/en/dev-1.x/user_guides/visualization.html)
- [Analysis Tools](https://mmselfsup.readthedocs.io/en/dev-1.x/user_guides/analysis_tools.html)

[Advanced Guides](https://mmselfsup.readthedocs.io/en/dev-1.x/advanced_guides/index.html) and [Colab Tutorials](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/demo/mmselfsup_colab_tutorial.ipynb) are also provided.

Please refer to [FAQ](https://mmselfsup.readthedocs.io/en/dev-1.x/notes/faq.html) for frequently asked questions.

## Model Zoo

Please refer to [Model Zoo.md](https://mmselfsup.readthedocs.io/en/dev-1.x/model_zoo.html) for a comprehensive set of pre-trained models and benchmarks.

Supported algorithms:

- [x] [Relative Location (ICCV'2015)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/relavive_loc)
- [x] [Rotation Prediction (ICLR'2018)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/rotation_pred)
- [x] [DeepCluster (ECCV'2018)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/deepcluster)
- [x] [NPID (CVPR'2018)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/npid)
- [x] [ODC (CVPR'2020)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/odc)
- [x] [MoCo v1 (CVPR'2020)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/mocov1)
- [x] [SimCLR (ICML'2020)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/simclr)
- [x] [MoCo v2 (arXiv'2020)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/byol)
- [x] [BYOL (NeurIPS'2020)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/mocov2)
- [x] [SwAV (NeurIPS'2020)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/swav)
- [x] [DenseCL (CVPR'2021)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/densecl)
- [x] [SimSiam (CVPR'2021)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/simsiam)
- [x] [Barlow Twins (ICML'2021)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/barlowtwins)
- [x] [MoCo v3 (ICCV'2021)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/mocov3)
- [x] [BEiT (ICLR'2022)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/beit)
- [x] [MAE (CVPR'2022)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/mae)
- [x] [SimMIM (CVPR'2022)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/simmim)
- [x] [MaskFeat (CVPR'2022)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/maskfeat)
- [x] [CAE (arXiv'2022)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/cae)
- [x] [MILAN (arXiv'2022)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/milan)
- [x] [BEiT v2 (arXiv'2022)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/beitv2)
- [x] [EVA (arXiv'2022)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/eva)
- [x] [MixMIM (ArXiv'2022)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/mixmim)

More algorithms are in our plan.

## Benchmark

| Benchmarks                                         | Setting                                                                                                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ImageNet Linear Classification (Multi-head)        | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
| ImageNet Linear Classification (Last)              |                                                                                                                                                                      |
| ImageNet Semi-Sup Classification                   |                                                                                                                                                                      |
| Places205 Linear Classification (Multi-head)       | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
| iNaturalist2018 Linear Classification (Multi-head) | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
| PASCAL VOC07 SVM                                   | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
| PASCAL VOC07 Low-shot SVM                          | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
| PASCAL VOC07+12 Object Detection                   | [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)               |
| COCO17 Object Detection                            | [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)               |
| Cityscapes Segmentation                            | [MMSeg](configs/benchmarks/mmsegmentation/cityscapes/fcn_r50-d8_769x769_40k_cityscapes.py)                                                                           |
| PASCAL VOC12 Aug Segmentation                      | [MMSeg](configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py)                                                                               |

## Contributing

We appreciate all contributions improving MMSelfSup. Please refer to [Contribution Guides](https://mmselfsup.readthedocs.io/en/dev-1.x/notes/contribution_guide.html) for more details about the contributing guideline.

## Acknowledgement

MMSelfSup is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new algorithms.

MMSelfSup originates from OpenSelfSup, and we appreciate all early contributions made to OpenSelfSup. A few contributors are listed here: Xiaohang Zhan ([@XiaohangZhan](http://github.com/XiaohangZhan)), Jiahao Xie ([@Jiahao000](https://github.com/Jiahao000)), Enze Xie ([@xieenze](https://github.com/xieenze)), Xiangxiang Chu ([@cxxgtxy](https://github.com/cxxgtxy)), Zijian He ([@scnuhealthy](https://github.com/scnuhealthy)).

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

we used mmselfsup and mmseg for this project. Please cite them if you use them. 
@misc{mmselfsup2021,
    title={{MMSelfSup}: OpenMMLab Self-Supervised Learning Toolbox and Benchmark},
    author={MMSelfSup Contributors},
    howpublished={\url{https://github.com/open-mmlab/mmselfsup}},
    year={2021}
}

