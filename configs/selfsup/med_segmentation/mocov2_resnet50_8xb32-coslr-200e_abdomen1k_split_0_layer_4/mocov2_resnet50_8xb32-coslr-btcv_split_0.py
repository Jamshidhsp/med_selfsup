_base_ = [
    '../../_base_/models/mocov2.py',
    '../../_base_/datasets/abdomen_1k.py',
    '../../_base_/schedules/sgd_coslr-200e_btcv.py',
    '../../_base_/default_runtime.py',
]


train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        ann_file='/home/hassanpour/datasets/abdomen_1k/img_dir/train/train_abdomen_ssl.txt',
    
    )
)





model = dict(
    data_preprocessor=dict( 
    # mean=[884.5, 884.5, 884.5],
    # std=[983.7, 983.7, 983.7]),
    mean= [0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]),
    backbone=dict(
        init_cfg=dict(
        type='Pretrained',
        checkpoint='/home/hassanpour/med_segmentation/checkpoints/resnet50-vissl.pth'

        ),
    ),
)


# # only keeps the latest 3 checkpoints
# default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))
