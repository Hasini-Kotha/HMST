"""
training/optimizers.py
----------------------
Standalone implementations of modern and standard optimizers for HMST:
- Lion: EvoLved Sign Momentum (Chen et al., Google Brain 2023)
- SAM: Sharpness-Aware Minimization wrapper (Foret et al., 2020)
- AdamW, Adam, SGD (built-in PyTorch)
- build_optimizer: Unified factory function
"""

import math
import torch
from torch.optim.optimizer import Optimizer
from typing import Dict, Any, Iterable, Union


# ─────────────────────────────────────────────────────────────────────────────
# 1. Lion Optimizer (Google Brain, 2023)
# ─────────────────────────────────────────────────────────────────────────────

class Lion(Optimizer):
    """
    Lion (EvoLved Sign Momentum) optimizer.
    Discovered by Google Brain (2023): https://arxiv.org/abs/2302.06675

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-4, usually 3-10x smaller than AdamW)
        betas: Coefficients for computing running gradient averages (default: (0.9, 0.99))
        weight_decay: Weight decay coefficient (default: 0.0, usually higher than AdamW)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # State initialization: tracking single momentum buffer
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]

                # Perform decoupled weight decay
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # Step 1: Update using sign of momentum
                # c_t = beta1 * exp_avg + (1 - beta1) * grad
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1).sign_()
                p.data.add_(update, alpha=-lr)

                # Step 2: Decay momentum buffer
                # exp_avg = beta2 * exp_avg + (1 - beta2) * grad
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


# ─────────────────────────────────────────────────────────────────────────────
# 2. SAM (Sharpness-Aware Minimization)
# ─────────────────────────────────────────────────────────────────────────────

class SAM(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer wrapper.
    Reference: Foret et al., 2020 (https://arxiv.org/abs/2010.01412)

    Finds parameters in flat minima regions, dramatically improving generalization
    on small vision datasets (like UWIQA).

    Args:
        params: Model parameters
        base_optimizer: Underlying optimizer class (e.g., torch.optim.AdamW or torch.optim.SGD)
        rho: Perturbation radius (default: 0.05)
        adaptive: If True, uses ASAM (Adaptive SAM)
        **kwargs: Arguments passed to the base optimizer (lr, weight_decay, etc.)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer: Any,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Step to the worst-case weight perturbation w + e(w)."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to local maximum

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Restore weights back to w and perform the base optimizer update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # restore original weights

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError(
            "SAM requires calling first_step() and second_step() across two forward/backward passes."
        )

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.norm(torch.stack(norms), p=2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Factory Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(
    model: torch.nn.Module,
    opt_name: str,
    lr: float,
    weight_decay: float,
    **kwargs,
) -> Optimizer:
    """
    Build an optimizer by name.

    Supports:
        - "adamw"
        - "adam"
        - "lion"
        - "sgd"
        - "sam_adamw"
        - "sam_sgd"
    """
    name = opt_name.lower().strip()
    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    elif name == "lion":
        return Lion(params, lr=lr, weight_decay=weight_decay)

    elif name == "sgd":
        momentum = kwargs.get("momentum", 0.9)
        nesterov = kwargs.get("nesterov", True)
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    elif name == "sam_adamw":
        rho = kwargs.get("rho", 0.05)
        return SAM(params, torch.optim.AdamW, rho=rho, lr=lr, weight_decay=weight_decay)

    elif name == "sam_sgd":
        rho = kwargs.get("rho", 0.05)
        momentum = kwargs.get("momentum", 0.9)
        return SAM(
            params,
            torch.optim.SGD,
            rho=rho,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {opt_name}. Supported: ['adamw', 'adam', 'lion', 'sgd', 'sam_adamw', 'sam_sgd']"
        )
