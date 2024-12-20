# dataset settings
dataset_type = 'mmcls.CustomDataset'
data_root = '/home2020/home/icube/jhassapo/swinUNETR_dataset/totalSegmentator/png_files/'
file_client_args = dict(backend='disk')
img_norm_cfg=dict(mean=[0], std=[1])
# The difference between mocov2 and mocov1 is the transforms in the pipeline
view_pipeline = [
    dict(
        type='RandomResizedCrop', size=512, scale=(0.2, 1.), backend='pillow'),
    # dict(
    #     type='RandomApply',
    #     transforms=[
    #         dict(
    #             type='ColorJitter',
    #             brightness=0.4,
    #             contrast=0.4,
    #             saturation=0.4,
    #             hue=0.1)
    #     ],
    #     prob=0.8),
    # dict(
    #     type='RandomGrayscale',
    #     prob=0.2,
    #     keep_channels=True,
    #     channel_weights=(0.114, 0.587, 0.2989)),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.5),
    dict(type='RandomFlip', prob=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile_v1', **img_norm_cfg),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='mig_dir/train/'),
        pipeline=train_pipeline))
