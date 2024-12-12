_base_ = [
    '../../_base_/models/mocov2.py',
    '../../_base_/datasets/total_segmentator_mocov2.py',
    '../../_base_/schedules/sgd_coslr-200e_in1k.py',
    '../../_base_/default_runtime.py',
]


train_dataloader = dict(
    dataset=dict(
        ann_file='total_segmentator_splits/split_0/train.txt',
    )
)

# only keeps the latest 3 checkpoints
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))
