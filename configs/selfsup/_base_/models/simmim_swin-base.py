# model settings
model = dict(
    type='SimMIM',
    data_preprocessor=dict(
        mean=[884.5, 884.5, 884.5],
        std=[983.7, 983.7, 983.7],
        bgr_to_rgb=True),
    backbone=dict(
        type='SimMIMSwinTransformer',
        arch='B',
        img_size=192,
        # img_size=512,
        stage_cfgs=dict(block_cfgs=dict(window_size=6))),
    neck=dict(type='SimMIMNeck', in_channels=128 * 2**3, encoder_stride=32),
    head=dict(
        type='SimMIMHead',
        patch_size=4,
        loss=dict(type='SimMIMReconstructionLoss', encoder_in_channels=3)))
