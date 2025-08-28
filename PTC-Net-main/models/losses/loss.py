# Code adapted or modified from MinkLoc3DV2 repo: https://github.com/jac99/MinkLoc3Dv2
# Warsaw University of Technology

import torch
from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.distances import LpDistance
from misc.utils import TrainingParams
from models.losses.loss_utils import *
from models.losses.truncated_smoothap import TruncatedSmoothAP


def make_losses(params: TrainingParams, dis_tensor=None):
    if params.loss == 'batchhardtripletmarginloss':
        return BatchHardTripletLossWithMasks(params.margin)
    elif params.loss == 'batchhardcontrastiveloss':
        return BatchHardContrastiveLossWithMasks(params.pos_margin, params.neg_margin)
    elif params.loss == 'truncatedsmoothap':
        return TruncatedSmoothAP(tau1=params.tau1, similarity=params.similarity,
                                 positives_per_query=params.positives_per_query)
    else:
        raise NotImplementedError(f"Unknown loss: {params.loss}")


class HardTripletMinerWithMasks:
    """Mine (a, p, n) using masks and keep simple distance stats."""
    def __init__(self, distance):
        self.distance = distance
        self.reset_stats()

    def reset_stats(self):
        self.max_pos_pair_dist = 0.0
        self.max_neg_pair_dist = 0.0
        self.mean_pos_pair_dist = 0.0
        self.mean_neg_pair_dist = 0.0
        self.min_pos_pair_dist = 0.0
        self.min_neg_pair_dist = 0.0

    def __call__(self, embeddings, positives_mask, negatives_mask):
        assert embeddings.dim() == 2
        with torch.no_grad():
            a, p, n = self.mine(embeddings, positives_mask, negatives_mask)
        return a, p, n

    def mine(self, embeddings, positives_mask, negatives_mask):
        self.reset_stats()
        # pairwise distance matrix
        dist_mat = self.distance(embeddings)  # [N, N]
        # hardest pos/neg per anchor (row-wise)
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = get_max_per_row(dist_mat, positives_mask)
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = get_min_per_row(dist_mat, negatives_mask)
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0), device=embeddings.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]

        # stats for logging
        if a.numel() > 0:
            pos_d = dist_mat[a, p]
            neg_d = dist_mat[a, n]
            self.mean_pos_pair_dist = pos_d.mean().item()
            self.mean_neg_pair_dist = neg_d.mean().item()
            self.max_pos_pair_dist = pos_d.max().item()
            self.max_neg_pair_dist = neg_d.max().item()
            self.min_pos_pair_dist = pos_d.min().item()
            self.min_neg_pair_dist = neg_d.min().item()
        return a, p, n


def get_max_per_row(mat, mask):
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat, mask):
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows


class BatchHardTripletLossWithMasks:
    def __init__(self, margin: float):
        self.margin = margin
        self.distance = LpDistance(normalize_embeddings=False, collect_stats=True)
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        # Reducer는 평균만 담당하게 두고, 통계는 우리가 직접 계산합니다.
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=False)
        self.loss_fn = losses.TripletMarginLoss(
            margin=self.margin, swap=True, distance=self.distance,
            reducer=reducer_fn, collect_stats=False
        )

    def __call__(self, embeddings, positives_mask, negatives_mask):
        a, p, n = self.miner_fn(embeddings, positives_mask, negatives_mask)

        # PML에 pre-mined indices를 넘겨서 손실 계산
        dummy_labels = torch.arange(embeddings.shape[0], device=embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, (a, p, n))

        # ---- 통계 직접 계산 (triplet hinge > 0 개수 등) ----
        dist_mat = self.distance(embeddings)  # 업데이트 겸 재사용
        if a.numel() > 0:
            d_ap = dist_mat[a, p]
            d_an = dist_mat[a, n]
            hinge = d_ap - d_an + self.margin
            num_non_zero_triplets = int((hinge > 0).sum().item())
        else:
            num_non_zero_triplets = 0

        stats = {
            'loss': float(loss.item()),
            'avg_embedding_norm': float(embeddings.norm(dim=1).mean().item()),
            'num_non_zero_triplets': num_non_zero_triplets,
            'num_triplets': int(a.numel()),
            'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
            'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
            'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
            'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
            'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
            'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
        }
        return loss, stats


class BatchHardContrastiveLossWithMasks:
    def __init__(self, pos_margin: float, neg_margin: float):
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.distance = LpDistance(normalize_embeddings=False, collect_stats=True)
        self.miner_fn = HardTripletMinerWithMasks(distance=self.distance)
        reducer_fn = reducers.AvgNonZeroReducer(collect_stats=False)
        self.loss_fn = losses.ContrastiveLoss(
            pos_margin=self.pos_margin, neg_margin=self.neg_margin,
            distance=self.distance, reducer=reducer_fn, collect_stats=False
        )

    def __call__(self, embeddings, positives_mask, negatives_mask):
        a, p, n = self.miner_fn(embeddings, positives_mask, negatives_mask)
        dummy_labels = torch.arange(embeddings.shape[0], device=embeddings.device)
        loss = self.loss_fn(embeddings, dummy_labels, (a, p, n))

        # ---- 통계 직접 계산 (위배된 pos/neg 페어 개수 및 각 파트 손실의 평균값) ----
        dist_mat = self.distance(embeddings)
        if a.numel() > 0:
            d_pos = dist_mat[a, p]
            d_neg = dist_mat[a, n]
            # 위배된 페어(손실 기여) 카운트
            pos_pairs_above_threshold = int((d_pos > self.pos_margin).sum().item())
            neg_pairs_above_threshold = int((d_neg < self.neg_margin).sum().item())  # '위배' 음수 페어 수
            # 로깅용 파트별 손실(정확한 값과 100% 일치할 필요는 없음; 통계용)
            pos_loss_val = torch.clamp(d_pos - self.pos_margin, min=0).pow(2).mean().item()
            neg_loss_val = torch.clamp(self.neg_margin - d_neg, min=0).pow(2).mean().item()
            num_pairs = int(2 * a.numel())
        else:
            pos_pairs_above_threshold = 0
            neg_pairs_above_threshold = 0
            pos_loss_val = 0.0
            neg_loss_val = 0.0
            num_pairs = 0

        stats = {
            'loss': float(loss.item()),
            'avg_embedding_norm': float(embeddings.norm(dim=1).mean().item()),
            'pos_pairs_above_threshold': pos_pairs_above_threshold,
            'neg_pairs_above_threshold': neg_pairs_above_threshold,
            'pos_loss': float(pos_loss_val),
            'neg_loss': float(neg_loss_val),
            'num_pairs': num_pairs,
            'mean_pos_pair_dist': self.miner_fn.mean_pos_pair_dist,
            'mean_neg_pair_dist': self.miner_fn.mean_neg_pair_dist,
            'max_pos_pair_dist': self.miner_fn.max_pos_pair_dist,
            'max_neg_pair_dist': self.miner_fn.max_neg_pair_dist,
            'min_pos_pair_dist': self.miner_fn.min_pos_pair_dist,
            'min_neg_pair_dist': self.miner_fn.min_neg_pair_dist
        }
        return loss, stats
