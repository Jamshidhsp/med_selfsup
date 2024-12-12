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



## Get Started

Download Abdomen-1K dataset (2D slices):
wget www.google.com
Download BTCV dataset (2D slices)
!wget www.google.com



## Installation
Having [Anaconda3][https://www.anaconda.com/products/individual#linux] installed follow the following steps for installation.
```
git clone https://github.com/CAMMA-public/SelfSupSurg



## Citation

we used mmselfsup and mmseg for this project. Please cite them if you use them. 
@misc{mmselfsup2021,
    title={{MMSelfSup}: OpenMMLab Self-Supervised Learning Toolbox and Benchmark},
    author={MMSelfSup Contributors},
    howpublished={\url{https://github.com/open-mmlab/mmselfsup}},
    year={2021}
}

