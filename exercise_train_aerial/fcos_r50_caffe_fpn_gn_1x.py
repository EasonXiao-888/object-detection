model = dict(
    type='FCOS',
    pretrained='open-mmlab://resnet50_caffe',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=81,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128]))
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    smoothl1_beta=0.11,
    gamma=2.0,
    alpha=0.25,
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
dataset_type = 'dota1_5'
data_root = 'data/dota1_5-split-1024'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='dota1_5',
        ann_file=
        'data/dota1_5-split-1024/trainval1024/DOTA1_5_trainval1024.json',
        img_prefix='data/dota1_5-split-1024/trainval1024/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    val=dict(
        type='dota1_5',
        ann_file=
        'data/dota1_5-split-1024/trainval1024/DOTA1_5_trainval1024.json',
        img_prefix='data/dota1_5-split-1024/trainval1024/images',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    test=dict(
        type='dota1_5',
        ann_file='data/dota1_5-split-1024/test1024/DOTA1_5_test1024.json',
        img_prefix='data/dota1_5-split-1024/test1024/images',
        img_scale=(1333, 800),
        img_norm_cfg=dict(
            mean=[102.9801, 115.9465, 122.7717],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=12)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
total_epochs = 12
device_ids = range(0, 4)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'exercise_train_aerial'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 1)
