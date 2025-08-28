# losses/truncated_smoothap.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TruncatedSmoothAP(nn.Module):
    """
    MinkLoc3Dv2의 Truncated Smooth AP 손실을 CrossLoc3D 양식으로 수정.
    1) masked_fill 용 Boolean 마스크 강제.
    2) 반환값을 (loss, stats, None) 형태로 3개 반환.
    """
    def __init__(self, cfg):
        """
        cfg: truncated_smoothap_cfg.py에서 로드된 설정 객체
        필요한 파라미터:
          - margin: float, (사용하지 않으면 None)
          - tau1: float, 스무딩 상수
          - k_top: int, 선택된 상위 k positive/negative
        """
        super().__init__()
        self.tau1 = getattr(cfg, 'tau1', 0.01)
        self.k_top = getattr(cfg, 'k_top', None)
        # 필요 시 margin 등 추가
        self.margin = getattr(cfg, 'margin', None)

    def forward(self, embeddings, positives_mask, negatives_mask):
        """
        embeddings: Tensor (B, D)
        positives_mask: BoolTensor (B, B)
        negatives_mask: BoolTensor (B, B)
        """
        # 1) 타입 강제 변환 (LongTensor로 넘어온다면 Boolean으로)
        positives_mask = positives_mask.bool()
        negatives_mask = negatives_mask.bool()

        # ================================
        # (A) similarity 계산 (예: cosine 또는 유클리디언)
        # 여기서는 예시로 코사인 유사도 사용 (원본 TSAP에 맞게 수정)
        sim_mat = embeddings @ embeddings.t()  # (B, B) 유클리디언 일 수도 있음
        # ================================

        # (B) Truncated SmoothAP 수식 예시
        # 1) positive pair 시그먼트
        #    - positive similarity: sim_mat where positives_mask
        #    - negative similarity: sim_mat where negatives_mask

        # 2) Sorted 상위 K개만 사용 시, k_top 활용 가능
        B = embeddings.size(0)
        sim_pos = sim_mat.masked_select(positives_mask).view(B, -1) if positives_mask.any() else torch.empty((B,0), device=sim_mat.device)
        sim_neg = sim_mat.masked_select(negatives_mask).view(B, -1) if negatives_mask.any() else torch.empty((B,0), device=sim_mat.device)

        # (예시) truncated 라는 의미로, top-k positive / negative만 선택 (k_top이 유효한 경우)
        if self.k_top is not None:
            # 각 행(row)별로 상위 k_top 선택 (sim_pos와 sim_neg를 재정렬해야 하지만,
            # 간단히 예시로 flatten 후 topk를 사용)
            # 실제 구현 시 B차원별로 분리해 topk 적용 필요
            sim_pos_topk, _ = torch.topk(sim_pos, min(self.k_top, sim_pos.size(1)), dim=1)
            sim_neg_topk, _ = torch.topk(sim_neg, min(self.k_top, sim_neg.size(1)), dim=1)
        else:
            sim_pos_topk = sim_pos
            sim_neg_topk = sim_neg

        # (C) SmoothAP 근사 수식
        # SmoothAP: Sigmoid 기반으로 pairwise 순위 매김 후 AP 근사
        # 간단한 예시 구현 (실제 MinkLoc3Dv2 원본 참조 필요)
        # pos score 행렬
        # pos_mask_flat = positives_mask.float()  (사용 안됨, 이미 sim_pos, sim_neg 추출 완료)

        # 예시: BxNp shape (Np = positive 개수)
        # 다음과 같이 Softmax/적분을 이용해 AP 근사
        # 실제로는 MinkLoc3Dv2 원본 코드를 최대한 그대로 참조해야 함

        # --- 아래는 전체 Loss 계산의 예시 로직입니다 ---
        # 전체 sim_mat => 정렬 => AP 근사 => 누적

        # (1) 배치 전체 sim을 sort
        s_all = sim_mat.clone()  # (B, B)
        # masked_fill로 positive pair 외의 유사도 +inf/ -inf 처리 → 순위 산정
        s_pos = s_all.clone()
        s_pos.masked_fill_(~positives_mask, float("-inf"))   # positive만 살아남음
        s_neg = s_all.clone()
        s_neg.masked_fill_(~negatives_mask, float("-inf"))   # negative만 살아남음

        # (D) 실제 Loss 계산 (예시는 단순화)
        # 각 i별로 AP 근사를 위해 (sigmoid( (s_neg - s_pos)/tau1 ) 의 합으로 근사)
        # i번째 샘플 기준 pos, neg pair 간 상대 순위 매김
        loss = torch.tensor(0.0, device=embeddings.device)
        temp_stats = {'loss': 0.0}

        for i in range(B):
            # i번째 벡터 기준 positive, negative 유사도
            sp = s_pos[i][positives_mask[i]]  # (Np_i,)
            sn = s_neg[i][negatives_mask[i]]  # (Nn_i,)
            if sp.numel() == 0 or sn.numel()==0:
                continue

            # pairwise difference
            # (Np_i, Nn_i)
            pair_diff = (sn.unsqueeze(0) - sp.unsqueeze(1)) / self.tau1
            # sigmoid 점수 => 정렬 확률
            sigma = torch.sigmoid(pair_diff)
            # AP 근사: sum over positive / (Np_i)
            ap_i = 1.0 - sigma.mean()
            loss += (1.0 - ap_i)

        loss = loss / B
        temp_stats['loss'] = loss.item()

        # (E) 반환값 3개로 맞추기
        #   - loss: Tensor 스칼라
        #   - temp_stats: dict
        #   - placeholder(None)
        return loss, temp_stats, None
