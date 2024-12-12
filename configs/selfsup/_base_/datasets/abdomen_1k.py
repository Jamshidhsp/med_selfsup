# dataset settings
dataset_type = 'mmcls.CustomDataset'
data_root = '/home/hassanpour/datasets/abdomen_1k/'
file_client_args = dict(backend='disk')
img_norm_cfg=dict(mean=[0], std=[1])
# The difference between mocov2 and mocov1 is the transforms in the pipeline

view_pipeline = [
    dict(
        type='RandomResizedCrop', size=336, scale=(0.2, 1.), backend='cv2'),
        # type='RandomResizedCrop', size=512, scale=(1., 1.), backend='cv2'),

    dict(
        type='RandAugment',
        policies='timm_increasing',
        # policies='debug',
        # policies='_RAND_INCREASING_TRANSFORMS',
        num_policies=2,
        total_level=10,
        magnitude_level=5,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
]

train_pipeline = [
    dict(type='LoadImageFromFile_v1', **img_norm_cfg),
    # dict(type='LoadImageFromFile'),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path']),
    

]

train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    drop_last=True,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='img_dir/train/'),
        pipeline=train_pipeline))
