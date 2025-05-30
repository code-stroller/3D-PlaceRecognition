loss_type = 'MultiSimilarityLoss'
loss_cfg = dict(
    alpha = 2.0,         # 양성 쌍 가중치 (기본값 2.0)
    beta = 50.0,         # 음성 쌍 가중치 (기본값 50.0)
    base = 0.5,          # 양/음성 임계값 용 오프셋 (기본값 0.5, 논문 Lambda)
    # normalize_embeddings = True  # (선택) 코사인 유사도 기반 학습을 위해 임베딩 정규화
)
