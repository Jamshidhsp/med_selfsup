# dataset settings
dataset_type = 'mmcls.ImageNet'
data_root = 'data/BTCV/img_dir/'
# data_root = 'data/imagenet/'
# file_client_args = dict(backend='disk', mean=[884.5], std=[983.7])
file_client_args = dict(backend='disk')
img_norm_cfg=dict(mean=[884.5], std=[983.7])

view_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    # dict(type='RandomGrayscale', prob=0.2, keep_channels=True),
    # dict(
    #     type='ColorJitter',
    #     brightness=0.4,
    #     contrast=0.4,
    #     saturation=0.4,
    #     hue=0.4),
    dict(type='RandomFlip', prob=0.5),
]

train_pipeline = [
    # dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadImageFromFile_v1', **img_norm_cfg),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    drop_last=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/train.txt',
        # ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))
