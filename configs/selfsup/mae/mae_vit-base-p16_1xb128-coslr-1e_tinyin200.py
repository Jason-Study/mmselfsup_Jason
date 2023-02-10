_base_ = 'mae_vit-base-p16_8xb512-coslr-400e_in1k.py'

# dataset
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        data_source=dict(
            data_prefix='data/tiny-imagenet-200/train',
            ann_file='data/tiny-imagenet-200/train.txt',
        )))

# optimizer
optimizer = dict(lr=1.5e-4 * 4096 / 256 * (32 / 512 * 8), )

runner = dict(max_epochs=1)

dist_params = dict(backend='gloo')  # windows"gloo", Linux"NCLL"
