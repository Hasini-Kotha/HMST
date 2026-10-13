"""
models/losses.py
----------------
Loss functions for HMST Architecture 1.

From §4.9 and solution3_architecture_analysis.md:

    L_total = λ₁·L_MOS + λ₂·L_rank + λ₃·L_cons

Where:
    L_MOS  = MSE loss between predicted and ground-truth quality (UWIQA/UUIQ)
           = MAE loss for UWIQA per architecture doc (λ₁·L_MAE)
    L_rank = Ranking loss for pairwise preference (SAUD native)
    L_cons = Cross-scale consistency loss (§4.9)
             Σ_{i,j} M_ij · (q_i^(32) − q_j^(128))²
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# L_MOS — MOS Regression Loss
# ─────────────────────────────────────────────────────────────────────────────

class MOSLoss(nn.Module):
    """
    Mean Absolute Error between predicted quality Q̂ and ground truth MOS y.

    UWIQA: L_MAE = (1/M) Σ |Q̂_n - y_n|
    SAUD:  L_MOS_Rank uses BT scores normalized to [0,1]
    """

    def __init__(self):
        super().__init__()

    def forward(self, Q_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Q_hat: [B]  predicted quality scores
            y:     [B]  ground truth MOS scores

        Returns:
            scalar loss
        """
        return F.l1_loss(Q_hat, y)


# ─────────────────────────────────────────────────────────────────────────────
# L_rank — Ranking Loss
# ─────────────────────────────────────────────────────────────────────────────

class RankingLoss(nn.Module):
    """
    Pairwise ranking loss for preference-based labels.

    Supports two modes:
    1. Pair mode (SAUD): explicitly fed (Q_a, Q_b, label)
       where label=1 means a > b
    2. Score mode (UWIQA): all pairs within the batch derived
       from quality scores y

    For SAUD training with return_pairs=True, pair mode is used.
    For UWIQA training, batch-level score mode is used.

    Uses margin-based ranking: max(0, margin - (Q_a - Q_b)) when a > b
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        Q_hat: torch.Tensor,
        y: torch.Tensor,
        Q_hat_b: Optional[torch.Tensor] = None,
        y_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score mode (UWIQA): creates all B*(B-1)/2 pairs within the batch.

        Args:
            Q_hat: [B]  predicted quality
            y:     [B]  ground truth quality (used to determine rank order)
            Q_hat_b, y_b: None in score mode

        Returns:
            scalar ranking loss
        """
        B = Q_hat.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=Q_hat.device)

        # All pairs
        loss = torch.tensor(0.0, device=Q_hat.device)
        n_pairs = 0
        for i in range(B):
            for j in range(i + 1, B):
                if y[i] > y[j] + 1e-4:
                    # i should score higher than j
                    loss = loss + F.relu(self.margin - (Q_hat[i] - Q_hat[j]))
                    n_pairs += 1
                elif y[j] > y[i] + 1e-4:
                    loss = loss + F.relu(self.margin - (Q_hat[j] - Q_hat[i]))
                    n_pairs += 1

        if n_pairs == 0:
            return torch.tensor(0.0, device=Q_hat.device)
        return loss / n_pairs


class PairRankingLoss(nn.Module):
    """
    Pair-mode ranking loss for SAUD pairwise preference data.

    L_rank = E[-log σ(Q_a - Q_b)]  when a is preferred over b
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        Q_a: torch.Tensor,   # [B]
        Q_b: torch.Tensor,   # [B]
        label: torch.Tensor, # [B]  1 if a > b, 0 otherwise
    ) -> torch.Tensor:
        """
        Returns:
            scalar loss
        """
        # When label=1, a should have higher score than b
        # BCE loss on sigmoid(Q_a - Q_b)
        diff = Q_a - Q_b  # [B]
        loss = F.binary_cross_entropy_with_logits(diff, label)
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# L_cons — Cross-Scale Consistency Loss (§4.9)
# ─────────────────────────────────────────────────────────────────────────────

class CrossScaleConsistencyLoss(nn.Module):
    """
    §4.9 — Cross-scale consistency constraint.

    L_cons = Σ_{i,j} M_ij · (q_i^(32) − q_j^(128))²

    Where M_ij = 1 when fine-scale patch i belongs to coarse-scale region j.

    Args:
        M: [N_fine, N_coarse]  precomputed correspondence matrix
           (built once and registered as a buffer)
    """

    def __init__(self, M: torch.Tensor):
        super().__init__()
        self.register_buffer("M", M)

    def forward(
        self,
        q_fine:   torch.Tensor,  # [B, N_fine, 1]  — p=32 quality scores
        q_coarse: torch.Tensor,  # [B, N_coarse, 1] — p=128 quality scores
    ) -> torch.Tensor:
        """
        Args:
            q_fine:   [B, N_fine, 1]
            q_coarse: [B, N_coarse, 1]

        Returns:
            scalar consistency loss
        """
        B = q_fine.shape[0]
        N_fine   = q_fine.shape[1]
        N_coarse = q_coarse.shape[1]

        # q_fine:   [B, N_fine]
        # q_coarse: [B, N_coarse]
        q_f = q_fine.squeeze(-1)    # [B, N_fine]
        q_c = q_coarse.squeeze(-1)  # [B, N_coarse]

        # For each (i, j) pair where M_ij = 1:
        # L_cons_sample = Σ M_ij (q_f_i - q_c_j)²
        # = Σ_j [ Σ_i M_ij (q_f_i - q_c_j)² ]

        # Efficient computation:
        # diff[i, j] = q_f[i] - q_c[j]  →  [B, N_fine, N_coarse]
        # weighted by M [N_fine, N_coarse]

        # diff: [B, N_fine, N_coarse]
        diff = q_f.unsqueeze(2) - q_c.unsqueeze(1)  # [B, N_fine, N_coarse]
        diff_sq = diff ** 2  # [B, N_fine, N_coarse]

        M = self.M  # [N_fine, N_coarse]
        loss = (M.unsqueeze(0) * diff_sq).sum() / (M.sum() * B + 1e-8)

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Combined HMST Loss
# ─────────────────────────────────────────────────────────────────────────────

class HMSTLoss(nn.Module):
    """
    L_total = λ₁·L_MOS + λ₂·L_rank + λ₃·L_cons

    From §4.9 and solution3_architecture_analysis.md.

    Args:
        M:            Correspondence matrix [N_fine, N_coarse] for L_cons
        lambda_mos:   λ₁
        lambda_rank:  λ₂
        lambda_cons:  λ₃
    """

    def __init__(
        self,
        M: torch.Tensor,
        lambda_mos:  float = 1.0,
        lambda_rank: float = 1.0,
        lambda_cons: float = 0.1,
        margin:      float = 0.1,
    ):
        super().__init__()
        self.lambda_mos  = lambda_mos
        self.lambda_rank = lambda_rank
        self.lambda_cons = lambda_cons
        self.margin      = margin

        self.l_mos   = MOSLoss()
        self.l_rank  = RankingLoss(margin=margin)
        self.l_cons  = CrossScaleConsistencyLoss(M)

    def forward(
        self,
        output: dict,
        y: torch.Tensor,
    ) -> dict:
        """
        Args:
            output: dict from HMST.forward()
            y:      [B] ground truth quality scores

        Returns:
            dict with keys:
                'loss':       total weighted loss
                'loss_mos':   L_MOS component
                'loss_rank':  L_rank component
                'loss_cons':  L_cons component
        """
        Q_hat      = output["Q_hat"]       # [B]
        qualities  = output["qualities"]   # list of [B, Ns, 1] per scale
        # qualities[0] = p=32 (fine), qualities[2] = p=128 (coarse)
        q_fine   = qualities[0]   # [B, N_fine, 1]
        q_coarse = qualities[2]   # [B, N_coarse, 1]

        loss_mos  = self.l_mos(Q_hat, y)
        loss_rank = self.l_rank(Q_hat, y)
        loss_cons = self.l_cons(q_fine, q_coarse)

        total = (
            self.lambda_mos  * loss_mos
            + self.lambda_rank * loss_rank
            + self.lambda_cons * loss_cons
        )

        return {
            "loss":      total,
            "loss_mos":  loss_mos,
            "loss_rank": loss_rank,
            "loss_cons": loss_cons,
        }
