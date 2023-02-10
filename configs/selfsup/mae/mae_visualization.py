model = dict(
    type='MAE',
    backbone=dict(type='MAEViT', arch='l', patch_size=16, mask_ratio=0.75),
    neck=dict(
        type='MAEPretrainDecoder',
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
    ),
    head=dict(type='MAEPretrainHead', norm_pix=True, patch_size=16))

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# dataset summary
data = dict(
    test=dict(pipeline=[
        dict(type='Resize', size=(224, 224)),
        dict(type='ToTensor'),
        dict(type='Normalize', **img_norm_cfg),
    ]))
