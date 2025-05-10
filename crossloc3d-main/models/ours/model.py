# models/ours/model.py

import torch
import torch.nn as nn
from .point_transformer_v3 import PointTransformerV3
from .netvlad import NetVladWrapper
from .pooling import MAC, SPoC, GeM

class PTV3BackboneWrapper(nn.Module):
    """
    Wraps PointTransformerV3 so that
      input:  (B, N, 3) raw point cloud
      output: (B, N, D) per-point feature
    """
    def __init__(self, cfg):
        super().__init__()
        # PTv3 init args 추출
        pt_keys = {
            'in_channels','order','stride','enc_depths','enc_channels',
            'enc_num_head','enc_patch_size','dec_depths','dec_channels',
            'dec_num_head','dec_patch_size','mlp_ratio','qkv_bias',
            'qk_scale','attn_drop','proj_drop','drop_path','pre_norm',
            'shuffle_orders','enable_rpe','enable_flash',
            'upcast_attention','upcast_softmax','cls_mode',
            'pdnorm_bn','pdnorm_ln','pdnorm_decouple','pdnorm_adaptive',
            'pdnorm_affine','pdnorm_conditions'
        }
        pt_cfg = {k: v for k, v in dict(cfg).items() if k in pt_keys}
        self.grid_size = cfg.grid_size
        self.model     = PointTransformerV3(**pt_cfg)

    def forward(self, x):
        # x: (B, N, 3)
        B, N, _ = x.shape
        coords = x.view(-1, 3)  # (B*N, 3)
        batch  = torch.arange(B, device=x.device).repeat_interleave(N)
        feats  = coords        # 입력 특징으로 좌표 사용
        data   = {
            'coord': coords,
            'grid_size': self.grid_size,
            'batch': batch,
            'feat': feats
        }
        out = self.model(data)   # Point dict 반환
        pf  = out['feat']        # (B*N, D)
        return pf.view(B, N, -1) # (B, N, D)

class Ours(nn.Module):
    """
    PTv3-backbone + pooling (NetVLAD / MAC / SPoC / GeM)
    """
    def __init__(self, cfg):
        super().__init__()
        assert cfg.backbone_cfg.type == 'PTV3'
        self.backbone = PTV3BackboneWrapper(cfg.backbone_cfg)

        pool_cfg     = cfg.pool_cfg
        # project 모듈은 일단 None으로 초기화
        self.project = None

        # pooling 타입에 따라 pool 모듈 선택
        if pool_cfg.type == 'NetVlad':
            self.pool = NetVladWrapper(pool_cfg)
            # Backbone 출력 차원 D
            D = cfg.backbone_cfg.dec_channels[0]
            # Pool 입력 차원 M
            M = pool_cfg.in_channels
            if D != M:
                # D→M 맞추기 위한 프로젝션
                C_mid = min(D, M) * 2
                self.project = nn.Sequential(
                    nn.Conv1d(D,   C_mid, kernel_size=1, bias=False),
                    nn.BatchNorm1d(C_mid),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(C_mid, M,    kernel_size=1, bias=False),
                    nn.BatchNorm1d(M),
                    nn.ReLU(inplace=True),
                )
        elif pool_cfg.type == 'MAC':
            self.pool = MAC(pool_cfg)
        elif pool_cfg.type == 'SPoC':
            self.pool = SPoC(pool_cfg)
        elif pool_cfg.type == 'GeM':
            self.pool = GeM(pool_cfg)
        else:
            raise ValueError(f"Unknown pool type {pool_cfg.type}")

    def forward(self, data):
        # data = (pcd_sparse_list, raw_pcd) 이지만 raw_pcd만 사용
        _, raw_pcd = data
        # raw_pcd: (B, N, 3)
        x = self.backbone(raw_pcd)   # -> (B, N, D)
        x = x.permute(0, 2, 1)        # -> (B, D, N)
        if self.project is not None:
            x = self.project(x)       # -> (B, M, N)
        return self.pool(x)           # -> (B, out_channels)
