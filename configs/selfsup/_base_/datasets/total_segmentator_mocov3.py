# dataset settings
dataset_type = 'mmcls.CustomDataset'
data_root = 'data/BTCV/img_dir/'
file_client_args = dict(backend='disk')
# img_norm_cfg=dict(mean=[884.5], std=[983.7])
img_norm_cfg=dict(mean=[0], std=[1])

# view_pipeline1 = [
#     dict(
#         type='RandomResizedCrop', size=224, scale=(0.2, 1.), backend='pillow'),
#     dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=1.),
#     dict(type='RandomSolarize', prob=0.),
#     dict(type='RandomFlip', prob=0.5),
# ]
# view_pipeline2 = [
#     dict(
#         type='RandomResizedCrop', size=224, scale=(0.2, 1.), backend='pillow'),
#     dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.1),
#     dict(type='RandomSolarize', prob=0.2),
#     dict(type='RandomFlip', prob=0.5),
# ]

train_pipeline = [
    dict(type='LoadImageFromFile_v1', file_client_args=file_client_args),
    dict(
        type='MultiView',
        num_views=[1, 1],
        transforms=[view_pipeline1, view_pipeline2]),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(
    # batch_size=256,
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))
