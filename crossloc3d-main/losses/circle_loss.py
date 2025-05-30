import torch
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity

class CircleLoss:
    """Circle Loss 구현 (pytorch-metric-learning 기반)"""
    def __init__(self, cfg):
        # 구성 파라미터 불러오기 (기본값: m=0.4, gamma=80)
        self.m = cfg.get('m', 0.4)
        self.gamma = cfg.get('gamma', 80)
        # CircleLoss는 Cosine 유사도를 기반으로 함 (normalize_embeddings는 강제로 True로 간주)
        self.distance = CosineSimilarity()  # 코사인 유사도 객체 (내부적으로 정규화된 코사인 유사도 사용)
        # pytorch-metric-learning의 CircleLoss 초기화
        self.loss_fn = losses.CircleLoss(m=self.m, gamma=self.gamma, distance=self.distance)
    
    def _mask_to_labels(self, positives_mask):
        """
        positives_mask (Bool Tensor)로부터 각 샘플의 클래스 레이블을 추론.
        같은 장소(양성 관계)에 속한 인덱스들을 동일 레이블로 설정.
        """
        device = positives_mask.device
        N = positives_mask.size(0)
        labels = torch.full((N,), -1, dtype=torch.long, device=device)
        current_label = 0
        for i in range(N):
            if labels[i] != -1:
                continue  # 이미 레이블 지정된 경우 skip
            # 새로운 클래스 레이블 할당
            labels[i] = current_label
            # i와 양성 관계인 모든 샘플에 동일 레이블 할당
            pos_indices = torch.nonzero(positives_mask[i], as_tuple=True)[0]
            labels[pos_indices] = current_label
            # 해당 양성들의 양성 관계도 동일 그룹으로 포함
            for j in pos_indices:
                labels[torch.nonzero(positives_mask[j], as_tuple=True)[0]] = current_label
            current_label += 1
        return labels
    
    def __call__(self, embeddings, positives_mask=None, negatives_mask=None):
        """
        embeddings: 특징 임베딩 텐서 (shape: [B, D])
        positives_mask, negatives_mask: 각 anchors에 대한 양성/음성 마스크 (Bool 텐서, shape: [B, B])
        """
        # 레이블 기반 손실 계산을 위해 mask로부터 레이블 생성
        if positives_mask is None:
            raise ValueError("CircleLoss requires positives_mask for label generation")
        labels = self._mask_to_labels(positives_mask)
        # 손실 값 계산 (pytorch-metric-learning CircleLoss는 labels 사용)
        loss = self.loss_fn(embeddings, labels)
        # 통계량 계산 (양성/음성 쌍 간 코사인 유사도 기반 거리)
        with torch.no_grad():
            # 코사인 유사도 행렬 계산
            # embeddings를 정규화하여 코사인 유사도 행렬 산출
            normed_emb = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            sim_mat = torch.mm(normed_emb, normed_emb.t())  # 각 샘플 간 코사인 유사도 [-1, 1]
            # 거리는 1 - 코사인유사도로 정의 (유사도 높을수록 거리 낮음)
            dist_mat = 1 - sim_mat 
            # 자기 자신 제외 (대각선 False 처리)
            pos_mask = positives_mask.clone().fill_diagonal_(False)
            neg_mask = negatives_mask.clone().fill_diagonal_(False) if negatives_mask is not None else ~pos_mask
            # 양성/음성 쌍 평균거리, 최소/최대거리 계산
            mean_pos = dist_mat[pos_mask].mean().item() if pos_mask.any() else 0.0
            mean_neg = dist_mat[neg_mask].mean().item() if neg_mask.any() else 0.0
            max_pos = dist_mat[pos_mask].max().item() if pos_mask.any() else 0.0
            min_pos = dist_mat[pos_mask].min().item() if pos_mask.any() else 0.0
            max_neg = dist_mat[neg_mask].max().item() if neg_mask.any() else 0.0
            min_neg = dist_mat[neg_mask].min().item() if neg_mask.any() else 0.0
            # 양성 쌍 개수 (대칭 쌍으로 count)
            pos_pair_count = pos_mask.sum().item()
        # 손실 및 통계 정보 딕셔너리 구성 (기존 구조와 호환되도록 키 유지)
        stats = {
            'loss': loss.item(),
            'mean_pos_pair_dist': mean_pos,
            'mean_neg_pair_dist': mean_neg,
            'max_pos_pair_dist': max_pos,
            'min_pos_pair_dist': min_pos,
            'max_neg_pair_dist': max_neg,
            'min_neg_pair_dist': min_neg,
            'num_non_zero_triplets': int(pos_pair_count),   # 양성 쌍 수를 임시 활용
            'num_triplets': int(pos_pair_count)            # (triplet이 아닌 경우에도 필드 유지)
        }
        return loss, stats, None  # triplet index에 해당하는 반환값은 None 처리
