# optimizer
optimizer = dict(type='SGD', lr=0.03, weight_decay=1e-6, momentum=0.9)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

# learning rate scheduler
param_scheduler = [
    dict(type='LinearLR', by_epoch=True, begin=5, end=500, convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300)
