_base_ = [
    '../../_base_/models/mocov2.py',
    '../../_base_/datasets/btcv_mocov2.py',
    '../../_base_/schedules/sgd_coslr-200e_in1k.py',
    '../../_base_/default_runtime.py',
]


train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        ann_file='/home/jamshid/Datasets/BTCV_png/btcv_splits/train_ssl.txt',
    
    )
)


model = dict(
    data_preprocessor=dict( 
    mean=[884.5, 884.5, 884.5],
    std=[983.7, 983.7, 983.7],
)
)

# # only keeps the latest 3 checkpoints
# default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))
