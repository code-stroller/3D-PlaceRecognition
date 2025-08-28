loss_type = 'LazyQuadrupletLoss'
loss_cfg = dict(
    alpha=0.5,               # margin for the anchor–negative term
    beta=0.2,                # margin for the negative–negative term
    normalize_embeddings=False
)
