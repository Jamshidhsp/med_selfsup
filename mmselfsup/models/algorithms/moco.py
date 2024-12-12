# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
from mmengine.model import ExponentialMovingAverage

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from .base import BaseModel

# 
import mmcv
import numpy as np


from mmcv.transforms import BaseTransform
import numbers
import random
import torch.nn.functional as F






def check_sequence_input(x: Sequence, name: str, req_sizes: tuple) -> None:
    """Check if the input is a sequence with the required sizes.

    Args:
        x (Sequence): The input sequence.
        name (str): The name of the input.
        req_sizes (tuple): The required sizes of the input.

    Returns:
        None
    """
    msg = req_sizes[0] if len(req_sizes) < 2 else ' or '.join(
        [str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError('{} should be a sequence of length {}.'.format(
            name, msg))
    if len(x) not in req_sizes:
        raise ValueError('{} should be sequence of length {}.'.format(
            name, msg))

class RandomCrop(BaseTransform):

    def __init__(self,
                 size: Union[int, Sequence[int]],
                 padding: Optional[Union[int, Sequence[int]]] = None,
                 pad_if_needed: bool = False,
                 pad_val: Union[numbers.Number, Sequence[numbers.Number]] = 0,
                 padding_mode: str = 'constant') -> None:
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.pad_val = pad_val
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img: np.ndarray, output_size: Tuple, num_regions) -> Tuple:

        height, width, channels= img.shape

    # Calculate the size of each square crop (assuming you want equal-sized square crops)
        crop_size = min(height, width) // num_regions

        # Initialize an empty list to store the crops
        crops_y = []
        crops_x = []

        # Iterate through the image to extract three non-overlapping square crops
        for i in range(num_regions):
        
            start_y = i * crop_size
            start_x = 0  # Always start from the leftmost column
            end_y = start_y + crop_size
            end_x = crop_size  # Always use the same width as the crop_size
            crops_y.append([start_y, end_y])
            crops_x.append([start_x, end_x])
        return crops_x, crops_y, crop_size

    def transform(self, img):
        
        results=[]
 
        # img = results['img']
        if self.padding is not None:
            img = mmcv.impad(img, padding=self.padding, pad_val=self.pad_val)

        # pad the height if needed
        if self.pad_if_needed and img.shape[0] < self.size[0]:
            img = mmcv.impad(
                img,
                padding=(0, self.size[0] - img.shape[0], 0,
                         self.size[0] - img.shape[0]),
                pad_val=self.pad_val,
                padding_mode=self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.shape[1] < self.size[1]:
            img = mmcv.impad(
                img,
                padding=(self.size[1] - img.shape[1], 0,
                         self.size[1] - img.shape[1], 0),
                pad_val=self.pad_val,
                padding_mode=self.padding_mode)

        # ymin, xmin, height, width = self.get_params(img, self.size)
        crops_x, crops_y, crop_size = self.get_params(img, self.size, num_regions)
        for i in range(num_regions):
            xmin, x_max = crops_x[i]
            ymin, y_max = crops_y[i]
            
            result = mmcv.imcrop(
                img, np.array([
                    xmin,
                    ymin,
                    xmin + crop_size - 1,
                    ymin + crop_size - 1,
                ]))
            results.append(result)
        return results

    def __repr__(self):
        return (self.__class__.__name__ +
                f'(size={self.size}, padding={self.padding})')



num_regions = 20


def feature_normalization(feature_map_z_1, feature_map_z_prime_1):

    feature_map_z_normalize = nn.functional.normalize(feature_map_z_1.reshape(feature_map_z_1.shape[0], feature_map_z_1.shape[1], -1), dim=2)
    feature_map_z_prime_normalize = nn.functional.normalize(feature_map_z_prime_1.reshape(feature_map_z_1.shape[0], feature_map_z_1.shape[1], -1), dim=2)

    feature_map_z = feature_map_z_normalize.reshape(feature_map_z_1.shape)
    feature_map_z_prime = feature_map_z_prime_normalize.reshape(feature_map_z_prime_1.shape)

    return feature_map_z, feature_map_z_prime




@MODELS.register_module()
class MoCo(BaseModel):

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 queue_len: int = 65536,
                 feat_dim: int = 128,
                 momentum: float = 0.999,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        # create momentum model
        self.encoder_k = ExponentialMovingAverage(
            nn.Sequential(self.backbone, self.neck), 1 - momentum)
        
        # create the queue
        self.queue_len = queue_len
        self.register_buffer('queue', torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    def extract_feat(self, inputs: List[torch.Tensor],
                     **kwarg) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Tuple[torch.Tensor]: Backbone outputs.
        """
        x = self.backbone(inputs[0])
        return x
    



    def loss(self, inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        im_q = inputs[0]
        im_k = inputs[1]

        # jamshid was here
        # feature_map_z = self.backbone(im_q)[0]
        # feature_map_z_prime = self.backbone(im_k)[0]
        
        feature_map_z_1 = self.backbone.maxpool(im_q)
        feature_map_z_prime_1 = self.backbone.maxpool(im_k)
        
        feature_map_z, feature_map_z_prime = feature_normalization(feature_map_z_1, feature_map_z_prime_1)


        # test
        sample_class = RandomCrop(size=(2,2))
        
        feature_map_z_premute  = torch.mean(feature_map_z, dim=1).permute(1,2,0)
        feature_map_z_prime_premute  = torch.mean(feature_map_z_prime, dim=1).permute(1,2,0)
        # N_crop regions
        cropped_regions_z = sample_class.transform(feature_map_z_premute)
        cropped_regions_z_prime = sample_class.transform(feature_map_z_prime_premute)
        
        for i in range(len(cropped_regions_z)):
            cropped_regions_z[i] = cropped_regions_z[i].permute(2, 0, 1)
            cropped_regions_z_prime[i] = cropped_regions_z_prime[i].permute(2,0,1)
        
        # cropped_regions_z_stacked = torch.stack([cropped_regions_z[0].reshape(cropped_regions_z[0].shape[0], -1), cropped_regions_z[1].reshape(cropped_regions_z[0].shape[0], -1), cropped_regions_z[2].reshape(cropped_regions_z[0].shape[0], -1)], dim=1)
        # cropped_regions_z_prime_stacked = torch.stack([cropped_regions_z_prime[0].reshape(cropped_regions_z_prime[0].shape[0], -1), cropped_regions_z_prime[1].reshape(cropped_regions_z_prime[0].shape[0], -1), cropped_regions_z_prime[2].reshape(cropped_regions_z_prime[0].shape[0], -1)], dim=1)
        
        cropped_regions_z_stacked = torch.stack([tensor.reshape(tensor.shape[0], -1) for tensor in cropped_regions_z], dim=1)
        cropped_regions_z_prime_stacked = torch.stack([tensor.reshape(tensor.shape[0], -1) for tensor in cropped_regions_z_prime], dim=1)

        
        q = self.neck(self.backbone(im_q))[0]  # queries: NxC
        # that's the z we want to apply
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self.encoder_k.update_parameters(
                nn.Sequential(self.backbone, self.neck))

            # shuffle for making use of BN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)
            feature_map_z_prime = self.encoder_k.module[0](im_k)[0]

            k = self.encoder_k(im_k)[0]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = batch_unshuffle_ddp(k, idx_unshuffle)



        # print('feature_map_z_prime.shape\n', feature_map_z_prime.shape)
        # print('feature_map_z.shape\n', feature_map_z.shape)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])


        #regional loss

        l_pos_regional = torch.einsum('bnc, bnc->bn', [cropped_regions_z_stacked, cropped_regions_z_prime_stacked])
        l_neg_regional_temp = torch.einsum('bnc, bkc->bnk', [cropped_regions_z_stacked, cropped_regions_z_prime_stacked])
        non_diagonal = l_neg_regional_temp[:, ~np.eye(num_regions, dtype=bool)]
        # l_neg_regional = non_diagonal.reshape(2, 6) 
        l_neg_regional = non_diagonal.reshape(cropped_regions_z_stacked.shape[0], int((num_regions-1)*num_regions)) 
        # l_neg_regional = non_diagonal.reshape(cropped_regions_z_stacked.shape[0], 2*0) 

        # loss = self.head(l_pos, l_neg)
        loss_global = self.head(l_pos, l_neg)
        
        loss_local = self.head(l_pos_regional, l_neg_regional)
        print('loss_global, loss_local', loss_global.item(), loss_local.item())
        loss = loss_global+0.5*loss_local
        # loss = loss_global
        # update the queue
        self._dequeue_and_enqueue(k)

        losses = dict(loss=loss)
        
        return losses