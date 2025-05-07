_base_ = [
    './base_cfg.py',
    './dataset_cfgs/oxford_cfg.py'
]

task_type = 'ours_me'

# Optimizer settings
optimizer_type = 'Adam'
optimizer_cfg = dict(
    lr=2e-4,             # Learning rate (tunable)
    weight_decay=0,      # Weight decay (tunable)
    betas=(0.9, 0.999),  # Adam betas (usually fine)
)

# Learning rate scheduler
scheduler_type = 'MultiStepLR'
scheduler_cfg = dict(
    gamma=0.1,           # LR decay factor (tunable)
    milestones=(80, 120, 160)  # Decay epochs (tunable)
)

end_epoch = 10  # Total number of training epochs (tunable)

# Training data settings
train_cfg = dict(
    save_per_epoch=10,   # Checkpoint interval
    val_per_epoch=5,     # Validation interval
    batch_sampler_type='ExpansionBatchSampler',
    batch_sampler_cfg=dict(
        max_batch_size=32,             # Maximum batch size (tunable)
        batch_size_expansion_rate=1.4, # Expansion rate
        batch_expansion_threshold=0.7,  # Threshold
        batch_size=16,                 # Base batch size (tunable)
        shuffle=True,
        drop_last=True,
    ),
    num_workers=0,      # DataLoader workers (adjust for your GPU)
)

# Evaluation data settings
eval_cfg = dict(
    batch_sampler_cfg=dict(
        batch_size=32,    # Eval batch size (tunable)
        drop_last=False,
    ),
    num_workers=0,
    normalize_embeddings=False,  # Normalize before KD-Tree
)

# Model definition
model_type = 'Ours'
model_cfg = dict(
    backbone_cfg=dict(
        type='PTV3',             # Use PointTransformerV3 backbone
        in_channels=3,            # Input feature dim (XYZ)
        grid_size=0.01,           # Voxel size (tunable)
        # --- PTv3 architecture parameters (tunable) ---
        order=('z', 'hilbert'),   # Serialization orders
        stride=(2,2,2,2),
        enc_depths=(2,2,2,6,2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2,4,8,16,32),
        enc_patch_size=(1024,)*5,
        dec_depths=(2,2,2,2),
        dec_channels=(64, 64, 128, 256), # dec_channels[0] must be same channels size with pool_cft.in_channels
        dec_num_head=(4,4,8,16),
        dec_patch_size=(1024,)*4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=('ScanNet','S3DIS','Structured3D'),
    ),
    pool_cfg=dict(
        type='NetVlad',           # Pooling type: NetVlad/MAC/GeM
        in_channels=64,           # Must match backbone_cfg.dec_channels[0]
        out_channels=256,         # Final descriptor dim (tunable)
        cluster_size=32,          # For NetVlad: number of clusters (tunable)
        gating=True,
        add_bn=True,
    ),
    quantization_size=[0.01, 0.12, 0.2],  # For FPN path but unused here
)
