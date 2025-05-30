from .contrastive_loss import BatchHardContrastiveLossWithMasks
from .triplet_loss import BatchHardTripletLossWithMasks
from .circle_loss import CircleLoss                # 새로 추가
from .multi_similarity_loss import MultiSimilarityLoss  # 새로 추가

def create_loss_fn(loss_type, cfg):
    type2loss_fn = {
        'BatchHardTripletMarginLoss': BatchHardTripletLossWithMasks,
        'BatchHardContrastiveLoss': BatchHardContrastiveLossWithMasks,
        'CircleLoss': CircleLoss,
        'MultiSimilarityLoss': MultiSimilarityLoss
    }
    if loss_type not in type2loss_fn:
        raise KeyError(f"Unknown loss type: {loss_type}")
    return type2loss_fn[loss_type](cfg)
