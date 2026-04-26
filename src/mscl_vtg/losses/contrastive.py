"""Within-scale and cross-scale contrastive losses (Eqs. 11–12).

Within-scale (Eq. 11):
  For each layer l, positive set P(l) = target moment indices,
  negative set N(l) = random non-target indices (|N| = |P|).
  L_within = -Σ_l Σ_{i∈P} Σ_{j∈P,j≠i} log[ exp(z_i·z_j) / (exp(z_i·z_j) + Σ_{n∈N} exp(z_i·z_n)) ]

Cross-scale (Eq. 12):
  Anchors = P(0) (lowest level), positives = P(l) for l>0, negatives = N(l).
  L_cross = -Σ_{i∈P(0)} Σ_l Σ_{j∈P(l)} log[ exp(z^0_i·z^l_j) / (exp(z^0_i·z^l_j) + Σ_{n∈N(l)} exp(z^0_i·z^l_n)) ]
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _info_nce_term(
    anchor: torch.Tensor,       # (D,)
    positive: torch.Tensor,     # (D,)
    negatives: torch.Tensor,    # (N_neg, D)
    temperature: float = 1.0,
) -> torch.Tensor:
    """Single InfoNCE term: -log[ exp(a·p/τ) / (exp(a·p/τ) + Σ exp(a·n/τ)) ]"""
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    if negatives.numel() == 0:
        return torch.tensor(0.0, device=anchor.device)
    negatives = F.normalize(negatives, dim=-1)

    pos_sim = (anchor * positive).sum() / temperature
    neg_sims = (anchor @ negatives.T) / temperature        # (N_neg,)
    logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])   # (1 + N_neg,)
    # -log softmax[0]
    return -F.log_softmax(logits, dim=0)[0]


def within_scale_contrastive_loss(
    feats: list[torch.Tensor],          # [Z^1, ..., Z^L], each (B, T_l, D)
    pos_masks: list[torch.Tensor],      # per-level (B, T_l) bool
    valid_masks: list[torch.Tensor | None],  # per-level (B, T_l) float
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute within-scale contrastive loss across all pyramid levels.

    For efficiency, we compute per-sample, per-level.
    """
    loss = torch.tensor(0.0, device=feats[0].device)
    count = 0

    for l, (z_l, p_mask) in enumerate(zip(feats, pos_masks)):
        B, T_l, D = z_l.shape
        v_mask = valid_masks[l] if valid_masks[l] is not None else torch.ones(B, T_l, device=z_l.device)

        for b in range(B):
            pos_idx = (p_mask[b] & (v_mask[b] > 0.5)).nonzero(as_tuple=False).squeeze(-1)
            neg_pool = (~p_mask[b] & (v_mask[b] > 0.5)).nonzero(as_tuple=False).squeeze(-1)

            if len(pos_idx) < 2 or len(neg_pool) == 0:
                continue

            # Sample |N| = |P| negatives
            n_neg = min(len(pos_idx), len(neg_pool))
            neg_idx = neg_pool[torch.randperm(len(neg_pool), device=z_l.device)[:n_neg]]
            neg_feats = z_l[b, neg_idx]  # (n_neg, D)

            for i_idx in range(len(pos_idx)):
                anchor = z_l[b, pos_idx[i_idx]]
                for j_idx in range(len(pos_idx)):
                    if i_idx == j_idx:
                        continue
                    positive = z_l[b, pos_idx[j_idx]]
                    loss = loss + _info_nce_term(anchor, positive, neg_feats, temperature)
                    count += 1

    return loss / max(count, 1)


def cross_scale_contrastive_loss(
    feats: list[torch.Tensor],          # [Z^0, Z^1, ..., Z^L]
    pos_masks: list[torch.Tensor],      # per-level (B, T_l) bool, index 0 = level 0
    valid_masks: list[torch.Tensor | None],
    temperature: float = 1.0,
) -> torch.Tensor:
    """Cross-scale contrastive loss (Eq. 12).

    Anchors from level 0, positives/negatives from levels 1..L.
    """
    loss = torch.tensor(0.0, device=feats[0].device)
    count = 0
    L = len(feats) - 1  # number of higher levels

    z0 = feats[0]  # (B, T_0, D)
    B = z0.shape[0]
    v0 = valid_masks[0] if valid_masks[0] is not None else torch.ones(B, z0.shape[1], device=z0.device)

    for b in range(B):
        anchor_idx = (pos_masks[0][b] & (v0[b] > 0.5)).nonzero(as_tuple=False).squeeze(-1)
        if len(anchor_idx) == 0:
            continue

        for l in range(1, L + 1):
            z_l = feats[l]
            T_l = z_l.shape[1]
            v_l = valid_masks[l] if valid_masks[l] is not None else torch.ones(T_l, device=z_l.device)
            if v_l.dim() == 2:
                v_l = v_l[b]
            p_l = pos_masks[l][b]

            pos_idx = (p_l & (v_l > 0.5)).nonzero(as_tuple=False).squeeze(-1)
            neg_pool = (~p_l & (v_l > 0.5)).nonzero(as_tuple=False).squeeze(-1)

            if len(pos_idx) == 0 or len(neg_pool) == 0:
                continue

            n_neg = min(len(pos_idx), len(neg_pool))
            neg_idx = neg_pool[torch.randperm(len(neg_pool), device=z_l.device)[:n_neg]]
            neg_feats = z_l[b, neg_idx]

            for ai in anchor_idx:
                anchor = z0[b, ai]
                for pi in pos_idx:
                    positive = z_l[b, pi]
                    loss = loss + _info_nce_term(anchor, positive, neg_feats, temperature)
                    count += 1

    return loss / max(count, 1)
