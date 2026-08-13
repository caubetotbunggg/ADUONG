"""EMA and AEMA teacher updates."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def ema_update(
    teacher: nn.Module,
    student: nn.Module,
    alpha: float = 0.999,
    global_step: Optional[int] = None,
) -> None:
    """
    Update teacher parameters in-place with EMA from student.

    Args:
        teacher      : teacher model (requires_grad=False)
        student      : student model (being trained)
        alpha        : EMA decay. Higher = slower teacher change.
        global_step  : current iteration. If provided, applies warmup:
                       effective_alpha = min(alpha, (1 + step) / (10 + step))
                       This prevents the teacher from diverging too fast at
                       the very start when the student is still random.
    """
    if global_step is not None:
        # Warmup: alpha ramps from ~0.09 → target_alpha over first ~10k steps
        warmup_alpha = (1.0 + global_step) / (10.0 + global_step)
        alpha = min(alpha, warmup_alpha)

    with torch.no_grad():
        # Update learnable parameters
        t_params = dict(teacher.named_parameters())
        s_params = dict(student.named_parameters())

        for name, t_p in t_params.items():
            if name in s_params:
                t_p.data.mul_(alpha).add_(s_params[name].data, alpha=1.0 - alpha)

        # Update buffers (e.g. BatchNorm running_mean / running_var)
        t_bufs = dict(teacher.named_buffers())
        s_bufs = dict(student.named_buffers())

        for name, t_b in t_bufs.items():
            if name in s_bufs and t_b.is_floating_point():
                t_b.data.mul_(alpha).add_(s_bufs[name].data, alpha=1.0 - alpha)
            elif name in s_bufs and not t_b.is_floating_point():
                t_b.data.copy_(s_bufs[name].data)


def _strip_scores(targets: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    """Detector training APIs consume boxes/labels; scores are inference metadata."""
    stripped = []
    for target in targets:
        stripped.append({
            "boxes": target["boxes"].detach(),
            "labels": target["labels"].detach(),
        })
    return stripped


def _classification_loss(loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Return the classification term used to rank AEMA-responsive parameters.

    Torchvision FCOS exposes "classification"; Faster R-CNN exposes
    "loss_classifier". For other detector wrappers, fall back to summing all
    losses so AEMA remains usable.
    """
    for key in ("classification", "loss_classifier"):
        if key in loss_dict:
            return loss_dict[key]
    return sum(v for v in loss_dict.values())


def _set_batchnorm_eval(module: nn.Module) -> None:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


class AEMAUpdater:
    """
    DDT-style adaptive EMA updater.

    Every step accumulates abs(teacher gradient) from a target-domain
    classification loss. Every `update_interval` steps, parameters in the
    global top `top_ratio` of accumulated gradient magnitudes use fast_alpha;
    the rest use slow_alpha.
    """

    def __init__(
        self,
        fast_alpha: float = 0.997,
        slow_alpha: float = 0.9996,
        top_ratio: float = 0.10,
        update_interval: int = 2,
    ) -> None:
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        self.top_ratio = top_ratio
        self.update_interval = update_interval
        self.steps_since_update = 0
        self.grad_accum: Dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        self.steps_since_update = 0
        self.grad_accum = {}

    def state_dict(self) -> Dict:
        return {
            "fast_alpha": self.fast_alpha,
            "slow_alpha": self.slow_alpha,
            "top_ratio": self.top_ratio,
            "update_interval": self.update_interval,
            "steps_since_update": self.steps_since_update,
            "grad_accum": {k: v.detach().cpu() for k, v in self.grad_accum.items()},
        }

    def load_state_dict(self, state: Dict, device: torch.device) -> None:
        self.steps_since_update = int(state.get("steps_since_update", 0))
        self.grad_accum = {
            k: v.to(device=device)
            for k, v in state.get("grad_accum", {}).items()
            if isinstance(v, torch.Tensor)
        }

    def accumulate_and_maybe_update(
        self,
        teacher: nn.Module,
        student: nn.Module,
        images: torch.Tensor,
        pseudo_targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, float]:
        """
        Accumulate teacher gradients, then update teacher from student when due.
        """
        log: Dict[str, float] = {
            "aema_updated": 0.0,
            "aema_fast_fraction": 0.0,
            "aema_importance_loss": 0.0,
        }
        if images.numel() == 0 or not pseudo_targets:
            self.steps_since_update += 1
            if self.steps_since_update >= self.update_interval:
                log.update(self._apply_update(teacher, student))
            return log

        was_training = teacher.training
        requires_grad = [p.requires_grad for p in teacher.parameters()]
        for p in teacher.parameters():
            p.requires_grad_(True)
        teacher.train()
        teacher.apply(_set_batchnorm_eval)
        teacher.zero_grad(set_to_none=True)

        clean_targets = _strip_scores(pseudo_targets)
        loss_dict = teacher(images, clean_targets)
        importance_loss = _classification_loss(loss_dict)
        importance_loss.backward()

        with torch.no_grad():
            for name, param in teacher.named_parameters():
                if param.grad is None:
                    continue
                grad = param.grad.detach().abs()
                if name not in self.grad_accum:
                    self.grad_accum[name] = torch.zeros_like(grad)
                self.grad_accum[name].add_(grad)

        log["aema_importance_loss"] = float(importance_loss.detach().item())
        teacher.zero_grad(set_to_none=True)
        for param, old_requires_grad in zip(teacher.parameters(), requires_grad):
            param.requires_grad_(old_requires_grad)
        teacher.train(was_training)

        self.steps_since_update += 1
        if self.steps_since_update >= self.update_interval:
            log.update(self._apply_update(teacher, student))
        return log

    def _apply_update(self, teacher: nn.Module, student: nn.Module) -> Dict[str, float]:
        t_params = dict(teacher.named_parameters())
        s_params = dict(student.named_parameters())
        t_bufs = dict(teacher.named_buffers())
        s_bufs = dict(student.named_buffers())

        flat_grads = [
            grad.reshape(-1)
            for name, grad in self.grad_accum.items()
            if name in t_params and grad.numel() > 0
        ]
        if flat_grads:
            all_grads = torch.cat(flat_grads)
            k = max(1, int(all_grads.numel() * self.top_ratio))
            threshold = torch.topk(all_grads, k=k, largest=True).values[-1]
        else:
            threshold = None

        fast_elems = 0
        total_elems = 0
        with torch.no_grad():
            for name, t_param in t_params.items():
                if name not in s_params:
                    continue
                s_param = s_params[name]
                total_elems += t_param.numel()
                grad = self.grad_accum.get(name)
                if grad is None or threshold is None:
                    t_param.data.mul_(self.slow_alpha).add_(
                        s_param.data, alpha=1.0 - self.slow_alpha
                    )
                    continue

                mask = grad >= threshold
                fast_elems += int(mask.sum().item())
                slow_updated = t_param.data * self.slow_alpha + s_param.data * (1.0 - self.slow_alpha)
                fast_updated = t_param.data * self.fast_alpha + s_param.data * (1.0 - self.fast_alpha)
                t_param.data.copy_(torch.where(mask, fast_updated, slow_updated))

            for name, t_buf in t_bufs.items():
                if name not in s_bufs:
                    continue
                s_buf = s_bufs[name]
                if t_buf.is_floating_point():
                    t_buf.data.mul_(self.slow_alpha).add_(s_buf.data, alpha=1.0 - self.slow_alpha)
                else:
                    t_buf.data.copy_(s_buf.data)

        self.reset()
        return {
            "aema_updated": 1.0,
            "aema_fast_fraction": (fast_elems / total_elems) if total_elems else 0.0,
            "aema_param_elements": float(total_elems),
        }


def copy_student_to_teacher(teacher: nn.Module, student: nn.Module) -> None:
    """Hard copy student weights into teacher (alpha=0). Used for initialization."""
    with torch.no_grad():
        for t_p, s_p in zip(teacher.parameters(), student.parameters()):
            t_p.data.copy_(s_p.data)
        for t_b, s_b in zip(teacher.buffers(), student.buffers()):
            if t_b.is_floating_point():
                t_b.data.copy_(s_b.data)
            else:
                t_b.data.copy_(s_b.data)
