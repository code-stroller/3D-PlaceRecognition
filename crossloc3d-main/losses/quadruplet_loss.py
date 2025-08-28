import numpy as np
import torch
from pytorch_metric_learning.distances import LpDistance


def get_max_per_row(mat, mask):
    mask = mask.to(torch.bool)
    non_zero_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = 0
    return torch.max(mat_masked, dim=1), non_zero_rows


def get_min_per_row(mat, mask):
    mask = mask.to(torch.bool)
    non_inf_rows = torch.any(mask, dim=1)
    mat_masked = mat.clone()
    mat_masked[~mask] = float('inf')
    return torch.min(mat_masked, dim=1), non_inf_rows


def get_random_neg_per_row(mask):
    """
    For each anchor (row), randomly sample one negative index among those where mask[i] == True.
    Returns:
      rand_indices: LongTensor of shape (N,) with the sampled index or -1 if none,
      valid_mask: BoolTensor of shape (N,) indicating which rows had at least one negative.
    """
    mask = mask.to(torch.bool)
    rand_indices = []
    valid = []
    for row in mask:
        idxs = torch.nonzero(row).view(-1)
        if idxs.numel() > 0:
            # sample one at random
            choice = idxs[torch.randint(idxs.numel(), (1,)).item()]
            rand_indices.append(choice)
            valid.append(True)
        else:
            rand_indices.append(torch.tensor(-1, device=mask.device))
            valid.append(False)
    rand_indices = torch.stack(rand_indices)
    valid = torch.tensor(valid, device=mask.device, dtype=torch.bool)
    return rand_indices, valid


class HardQuadrupletMinerWithMasks:
    """
    Miner that for each anchor finds:
      - hardest positive (max distance among positives)
      - hardest negative (min distance among negatives)
      - a random negative (sampled among negatives)
    """
    def __init__(self, distance: LpDistance):
        self.distance = distance

    def __call__(self, embeddings, positives_mask, negatives_mask):
        return self.mine(embeddings, positives_mask, negatives_mask)

    def mine(self, embeddings, positives_mask, negatives_mask):
        dist_mat = self.distance(embeddings)  # (N, N) distance matrix

        # hardest positives
        (pos_dists, pos_idxs), pos_keep = get_max_per_row(dist_mat, positives_mask)
        # hardest negatives
        (neg_dists, neg_idxs), neg_keep = get_min_per_row(dist_mat, negatives_mask)
        # random negatives
        rand_idxs, rand_keep = get_random_neg_per_row(negatives_mask)

        # only keep anchors where all three exist
        keep_mask = pos_keep & neg_keep & rand_keep
        a_idx = torch.where(keep_mask)[0]

        a = a_idx
        p = pos_idxs[ a ]
        n = neg_idxs[ a ]
        rn = rand_idxs[ a ]

        return a, p, n, rn


class BatchHardQuadrupletLossWithMasks:
    """
    Lazy Quadruplet Loss:
      L = E_a [ max(α + d(a,p) - d(a,n), 0) + max(β + d(a,p) - d(n*, n), 0) ]
    where n* is a random negative.
    """
    def __init__(self, cfg):
        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.normalize_embeddings = cfg.normalize_embeddings
        self.distance = LpDistance(normalize_embeddings=self.normalize_embeddings)
        self.miner_fn = HardQuadrupletMinerWithMasks(distance=self.distance)

    def __call__(self, embeddings, positives_mask, negatives_mask):
        # mine quadruplets
        a, p, n, rn = self.miner_fn(embeddings, positives_mask, negatives_mask)

        # compute the three distances
        delta_pos = self.distance(embeddings[a], embeddings[p])
        delta_neg = self.distance(embeddings[a], embeddings[n])
        delta_neg2 = self.distance(embeddings[n], embeddings[rn])

        # two hinge terms
        loss1 = torch.clamp(self.alpha + delta_pos - delta_neg, min=0.0)
        loss2 = torch.clamp(self.beta  + delta_pos - delta_neg2, min=0.0)
        losses = loss1 + loss2

        # final scalar loss
        if losses.numel() > 0:
            loss = torch.mean(losses)
        else:
            loss = torch.tensor(0.0, device=embeddings.device)

        # some diagnostics
        stats = {
            'loss': loss.item(),
            'num_quadruplets': a.numel(),
            'num_non_zero_quadruplets': torch.sum(losses > 0).item(),
            'mean_pos_pair_dist': delta_pos.mean().item() if delta_pos.numel()>0 else 0.0,
            'mean_neg_pair_dist': delta_neg.mean().item() if delta_neg.numel()>0 else 0.0,
            'mean_neg2_pair_dist': delta_neg2.mean().item() if delta_neg2.numel()>0 else 0.0,
        }

        # return same-style quartet of indices for logging/debug
        return loss, stats, (a, p, n, rn)
