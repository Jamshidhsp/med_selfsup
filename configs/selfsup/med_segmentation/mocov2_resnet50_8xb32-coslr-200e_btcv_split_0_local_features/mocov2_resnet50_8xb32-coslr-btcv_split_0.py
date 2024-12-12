_base_ = [
    '../../_base_/models/mocov2.py',
    '../../_base_/datasets/btcv_mocov2.py',
    '../../_base_/schedules/sgd_coslr-200e_btcv.py',
    '../../_base_/default_runtime.py',
]


train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        ann_file='/home/jamshid/Datasets/BTCV_png/btcv_splits/train_ssl.txt',
    
    )
)



view_pipeline = [
    dict(
        type='RandomResizedCrop', size=512, scale=(0.2, 1.), backend='cv2'),

    dict(
        type='RandAugment',
        policies='local_features',
        # policies='debug',
        # policies='_RAND_INCREASING_TRANSFORMS',
        num_policies=2,
        total_level=10,
        magnitude_level=5,
        magnitude_std=0.5,
        hparams=dict(pad_val=[104, 116, 124], interpolation='bicubic')),
]


model = dict(
    data_preprocessor=dict( 
    mean=[884.5, 884.5, 884.5],
    std=[983.7, 983.7, 983.7]),
    # mean= [0.485, 0.456, 0.406],
    # std=[0.229, 0.224, 0.225]),
    backbone=dict(
        init_cfg=dict(
        type='Pretrained',
        checkpoint='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/med_segmentation/pretrain/resnet50-vissl.pth'
        # checkpoint = "/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/med_segmentation/runs/vissl/deeplabv3_plus_r50-d8_512x512_btcv_moco_1K_vissl_set_2_2/best_mDice_iter_16000.pth",
        # prefix="backbone."
        ),
    ),
)


# # only keeps the latest 3 checkpoints
# default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))
