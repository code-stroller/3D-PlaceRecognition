import torch
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity

class MultiSimilarityLoss:
    """Multi-Similarity Loss 구현 (pytorch-metric-learning 기반)"""
    def __init__(self, cfg):
        # 구성 파라미터 불러오기 (기본값: alpha=2, beta=50, base=0.5)
        self.alpha = cfg.get('alpha', 2.0)
        self.beta = cfg.get('beta', 50.0)
        self.base = cfg.get('base', 0.5)
        # MultiSimilarityLoss도 일반적으로 코사인 유사도 사용
        self.distance = CosineSimilarity()
        self.loss_fn = losses.MultiSimilarityLoss(alpha=self.alpha, beta=self.beta, base=self.base, distance=self.distance)
    
    def _mask_to_labels(self, positives_mask):
        """positives_mask로부터 클래스 레이블 생성 (CircleLoss와 동일)"""
        device = positives_mask.device
        N = positives_mask.size(0)
        labels = torch.full((N,), -1, dtype=torch.long, device=device)
        current_label = 0
        for i in range(N):
            if labels[i] != -1:
                continue
            labels[i] = current_label
            pos_indices = torch.nonzero(positives_mask[i], as_tuple=True)[0]
            labels[pos_indices] = current_label
            for j in pos_indices:
                labels[torch.nonzero(positives_mask[j], as_tuple=True)[0]] = current_label
            current_label += 1
        return labels
    
    def __call__(self, embeddings, positives_mask=None, negatives_mask=None):
        if positives_mask is None:
            raise ValueError("MultiSimilarityLoss requires positives_mask for label generation")
        labels = self._mask_to_labels(positives_mask)
        # pytorch-metric-learning MultiSimilarityLoss 계산 (labels 사용)
        loss = self.loss_fn(embeddings, labels)
        # 통계량 계산 (코사인 유사도 기반 유사 방식)
        with torch.no_grad():
            normed_emb = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            sim_mat = torch.mm(normed_emb, normed_emb.t())
            dist_mat = 1 - sim_mat
            pos_mask = positives_mask.clone().fill_diagonal_(False)
            neg_mask = negatives_mask.clone().fill_diagonal_(False) if negatives_mask is not None else ~pos_mask
            mean_pos = dist_mat[pos_mask].mean().item() if pos_mask.any() else 0.0
            mean_neg = dist_mat[neg_mask].mean().item() if neg_mask.any() else 0.0
            max_pos = dist_mat[pos_mask].max().item() if pos_mask.any() else 0.0
            min_pos = dist_mat[pos_mask].min().item() if pos_mask.any() else 0.0
            max_neg = dist_mat[neg_mask].max().item() if neg_mask.any() else 0.0
            min_neg = dist_mat[neg_mask].min().item() if neg_mask.any() else 0.0
            pos_pair_count = pos_mask.sum().item()
        stats = {
            'loss': loss.item(),
            'mean_pos_pair_dist': mean_pos,
            'mean_neg_pair_dist': mean_neg,
            'max_pos_pair_dist': max_pos,
            'min_pos_pair_dist': min_pos,
            'max_neg_pair_dist': max_neg,
            'min_neg_pair_dist': min_neg,
            'num_non_zero_triplets': int(pos_pair_count),
            'num_triplets': int(pos_pair_count)
        }
        return loss, stats, None
