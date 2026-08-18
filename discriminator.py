"""
Adversarial domain adaptation components (DANN-style).

GradientReversal      : reverses gradient sign × lambda during backprop
DomainDiscriminator   : binary domain classifier (domain A=0 vs B=1)
grl_lambda_schedule   : progressive lambda schedule from DANN paper
compute_adv_loss      : one-call adversarial loss used by the trainer
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (lambda_,) = ctx.saved_tensors
        return -lambda_ * grad, None


class GradientReversal(nn.Module):
    """
    No-op in forward; multiplies upstream gradient by -lambda_ in backward.
    Call set_lambda() each step to implement a progressive schedule.
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lam = torch.tensor(self.lambda_, dtype=x.dtype, device=x.device)
        return _GRLFunction.apply(x, lam)

    def set_lambda(self, value: float) -> None:
        self.lambda_ = value


# ---------------------------------------------------------------------------
# Domain Discriminator
# ---------------------------------------------------------------------------

class DomainDiscriminator(nn.Module):
    """
    2-layer MLP domain classifier (2 classes).

    Input  : [B, in_features] globally-pooled backbone features.
    Output : [B, 2] raw logits (use F.cross_entropy).

    Label convention: domain A → 0, domain B → 1.
    Architecture: in_features → hidden → 2  (one hidden layer, lightweight).
    """

    def __init__(self, in_features: int = 2048, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B, 2]


# ---------------------------------------------------------------------------
# Lambda schedule
# ---------------------------------------------------------------------------

def grl_lambda_schedule(
    step: int,
    phase_start: int,
    phase_end: int,
    max_lambda: float = 1.0,
) -> float:
    """
    DANN progressive schedule: lambda increases 0 → max_lambda during a phase.

    lambda(p) = max_lambda * (2 / (1 + exp(-10p)) - 1),  p ∈ [0, 1]

    Starts near 0 so early training is stable, then grows to max_lambda.
    """
    if phase_end <= phase_start:
        return max_lambda
    progress = max(0.0, min(1.0, (step - phase_start) / (phase_end - phase_start)))
    return max_lambda * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


# ---------------------------------------------------------------------------
# Adversarial loss
# ---------------------------------------------------------------------------

def compute_adv_loss(
    features:       torch.Tensor,
    discriminator:  DomainDiscriminator,
    grl:            GradientReversal,
    n_a:            int,
    label_smoothing: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Adversarial alignment loss for a mixed-domain batch.

    Args:
        features        : [B, D] globally-pooled backbone features WITH gradient.
        discriminator   : 2-class domain classifier.
        grl             : GradientReversal (lambda already set by caller).
        n_a             : samples from domain A (label=0); rest are domain B (label=1).
        label_smoothing : smoothing ε for CrossEntropyLoss (default 0.1).
                          Prevents discriminator from becoming over-confident,
                          keeping gradients alive even when acc is high.

    Returns:
        loss     : scalar tensor (unscaled; caller applies adv_weight).
        log_dict : {"adv_loss": float, "disc_acc": float}

    One backward() simultaneously:
      - updates discriminator (forward gradient → better domain classification)
      - updates student backbone (reversed gradient → domain-invariant features)
    """
    device = features.device
    B      = features.shape[0]

    domain_labels = torch.cat([
        torch.zeros(n_a,     dtype=torch.long, device=device),
        torch.ones( B - n_a, dtype=torch.long, device=device),
    ])

    logits = discriminator(grl(features))                              # [B, 2]
    loss   = F.cross_entropy(logits, domain_labels, label_smoothing=label_smoothing)

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == domain_labels).float().mean().item()

    return loss, {"adv_loss": loss.item(), "disc_acc": acc}
