loss_type = 'CircleLoss'
loss_cfg = dict(
    m = 0.3,             # CircleLoss margin (결정경계 여유 변수)
    gamma = 256,          # CircleLoss 스케일 계수
    # normalize_embeddings 항목은 명시하지 않아도 코사인 유사도 사용으로 자동 정규화됨
)
