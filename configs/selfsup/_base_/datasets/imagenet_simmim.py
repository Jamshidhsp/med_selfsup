# dataset settings
dataset_type = 'mmcls.ImageNet'
# data_root = 'data/imagenet/'
data_root = 'data/BTCV/img_dir/'
file_client_args = dict(backend='disk')
# img_norm_cfg=dict(mean=[884.5], std=[983.7])
img_norm_cfg=dict(mean=[0], std=[1])

train_pipeline = [
    # dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadImageFromFile_v1', **img_norm_cfg),
    dict(
        type='RandomResizedCrop',
        size=512,
        scale=(0.67, 1.0),
        ratio=(3. / 4., 4. / 3.)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='SimMIMMaskGenerator',
        input_size=512,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.6),
    dict(
        type='PackSelfSupInputs',
        algorithm_keys=['mask'],
        meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='meta/train.txt',
        ann_file='train/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))

# for visualization
vis_pipeline = [
    # dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadImageFromFile_v1', **img_norm_cfg),
    dict(type='Resize', scale=(512, 512), backend='pillow'),
    dict(
        type='SimMIMMaskGenerator',
        input_size=512,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.6),
    dict(
        type='PackSelfSupInputs',
        algorithm_keys=['mask'],
        meta_keys=['img_path'])
]
