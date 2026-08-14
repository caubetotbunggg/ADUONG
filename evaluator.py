"""
Detection Evaluator — mAP computation without pycocotools or torchmetrics.

Implements:
  - Per-class AP@IoU  (VOC 11-point interpolation)
  - mAP@0.5           (mean over classes)
  - mAP@0.5:0.95      (COCO-style, mean over IoU thresholds)
  - Per-domain tracking across curriculum phases

Dependencies: only torch + torchvision.ops.box_iou
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from adaptive_threshold import AdaptiveThresholdScheduler

logger = logging.getLogger(__name__)


def _unwrap(model):
    return model.module if isinstance(model, (nn.DataParallel,)) else model


# ---------------------------------------------------------------------------
# Core AP computation
# ---------------------------------------------------------------------------

def _compute_ap_voc11(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    """
    VOC 11-point interpolated AP.

    For each recall threshold t in {0.0, 0.1, ..., 1.0}:
      p(t) = max precision where recall >= t
    AP = mean(p(t))
    """
    ap = 0.0
    for t in torch.linspace(0.0, 1.0, 11):
        mask = recalls >= t
        ap += precisions[mask].max().item() if mask.any() else 0.0
    return ap / 11.0


def _compute_ap_auc(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    """
    Area-under-curve AP (COCO-style interpolation).
    Monotonically decreasing envelope then trapezoid integration.
    """
    # Sentinel values
    mrec = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])

    # Monotonically decreasing envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = torch.max(mpre[i], mpre[i + 1])

    # Integrate at points where recall changes
    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    ap  = ((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum().item()
    return ap


def _compute_class_ap(
    pred_list: List[Dict],    # per-image predictions for this class
    gt_list:   List[Dict],    # per-image ground-truths  for this class
    iou_thresh: float = 0.5,
    interp: str = "voc11",    # "voc11" | "auc"
) -> float:
    """
    Compute AP for a single class across all images.

    pred_list[i] = {"boxes": [N,4], "scores": [N]}
    gt_list[i]   = {"boxes": [M,4]}
    """
    n_gt = sum(len(g["boxes"]) for g in gt_list)
    if n_gt == 0:
        return float("nan")   # no GT for this class → skip in mAP average

    all_scores: List[float] = []
    all_tp:     List[int]   = []

    for pred, gt in zip(pred_list, gt_list):
        pb = pred["boxes"]   # [N, 4]
        ps = pred["scores"]  # [N]
        gb = gt["boxes"]     # [M, 4]

        if len(pb) == 0:
            continue

        # Sort predictions by score descending
        order = ps.argsort(descending=True)
        pb, ps = pb[order], ps[order]

        matched = torch.zeros(len(gb), dtype=torch.bool)

        for i in range(len(pb)):
            all_scores.append(ps[i].item())

            if len(gb) == 0:
                all_tp.append(0)
                continue

            ious        = box_iou(pb[i].unsqueeze(0), gb)[0]   # [M]
            best_iou, j = ious.max(0)

            if best_iou >= iou_thresh and not matched[j]:
                matched[j] = True
                all_tp.append(1)
            else:
                all_tp.append(0)

    if not all_scores:
        return 0.0

    # Sort all predictions across images by score
    order   = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
    tp_arr  = torch.tensor([all_tp[i] for i in order], dtype=torch.float32)

    cum_tp  = tp_arr.cumsum(0)
    cum_fp  = (1 - tp_arr).cumsum(0)
    recalls    = cum_tp / n_gt
    precisions = cum_tp / (cum_tp + cum_fp + 1e-9)

    fn = _compute_ap_voc11 if interp == "voc11" else _compute_ap_auc
    return fn(recalls, precisions)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class DetectionEvaluator:
    """
    Accumulates predictions and ground-truths, computes mAP.

    Workflow:
        evaluator.reset()
        for images, targets in val_loader:
            preds = model(images)
            evaluator.update(preds, targets)
        results = evaluator.compute()
    """

    def __init__(
        self,
        num_classes:  int,
        class_names:  Optional[List[str]] = None,
        iou_thresholds: Optional[List[float]] = None,
        interp: str = "voc11",
    ) -> None:
        """
        Args:
            num_classes    : number of foreground classes (0-indexed)
            class_names    : human-readable names for logging
            iou_thresholds : list of IoU thresholds for mAP computation
                             default = [0.5] → mAP@0.5
                             COCO-style = [0.5, 0.55, ..., 0.95]
            interp         : "voc11" or "auc"
        """
        self.num_classes     = num_classes
        self.class_names     = class_names or [f"cls_{i}" for i in range(num_classes)]
        self.iou_thresholds  = iou_thresholds or [0.5]
        self.interp          = interp
        self.reset()

    def reset(self) -> None:
        """Clear accumulated predictions and ground-truths."""
        # per_class_preds[c][img_idx] = {"boxes": T, "scores": T}
        self._preds: List[Dict] = []
        self._gts:   List[Dict] = []

    def update(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        targets:     List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Add one batch of predictions and ground-truths.

        predictions : output of model.eval()(images)
                      each dict: {"boxes":[N,4], "labels":[N], "scores":[N]}
        targets     : ground-truth dicts
                      each dict: {"boxes":[M,4], "labels":[M]}
        """
        for pred, target in zip(predictions, targets):
            self._preds.append({k: v.cpu() for k, v in pred.items() if isinstance(v, torch.Tensor)})
            self._gts.append(  {k: v.cpu() for k, v in target.items() if isinstance(v, torch.Tensor)})

    def compute(self) -> Dict[str, float]:
        """
        Compute mAP@IoU for all configured IoU thresholds.

        Returns dict with:
          "mAP@0.5"          : mean AP over classes at IoU=0.5
          "mAP@0.5:0.95"     : COCO-style (if multiple IoU thresholds)
          "AP@0.5/{class}"   : per-class AP at IoU=0.5
        """
        results: Dict[str, float] = {}

        # Per-class, per-image prediction/GT split
        by_cls_pred: Dict[int, List[Dict]] = defaultdict(lambda: [{}] * len(self._preds))
        by_cls_gt:   Dict[int, List[Dict]] = defaultdict(lambda: [{}] * len(self._gts))

        for img_i, (pred, gt) in enumerate(zip(self._preds, self._gts)):
            for c in range(self.num_classes):
                pm = pred["labels"] == c
                gm = gt["labels"]   == c

                by_cls_pred[c][img_i] = {
                    "boxes":  pred["boxes"][pm]  if pm.any() else torch.zeros(0, 4),
                    "scores": pred["scores"][pm] if pm.any() else torch.zeros(0),
                }
                by_cls_gt[c][img_i] = {
                    "boxes": gt["boxes"][gm] if gm.any() else torch.zeros(0, 4),
                }

        all_threshold_maps: List[float] = []

        for iou_t in self.iou_thresholds:
            class_aps: List[float] = []
            for c in range(self.num_classes):
                ap = _compute_class_ap(
                    by_cls_pred[c], by_cls_gt[c],
                    iou_thresh=iou_t,
                    interp=self.interp,
                )
                if not (ap != ap):   # skip NaN (class absent in GT)
                    class_aps.append(ap)
                    if iou_t == 0.5:
                        results[f"AP@0.5/{self.class_names[c]}"] = round(ap, 4)

            map_at_t = sum(class_aps) / len(class_aps) if class_aps else 0.0

            if iou_t == 0.5:
                results["mAP@0.5"] = round(map_at_t, 4)
            all_threshold_maps.append(map_at_t)

        if len(self.iou_thresholds) > 1:
            results["mAP@0.5:0.95"] = round(
                sum(all_threshold_maps) / len(all_threshold_maps), 4
            )

        results["num_images"] = len(self._preds)
        return results


# ---------------------------------------------------------------------------
# Phase evaluator — runs evaluation at curriculum phase transitions
# ---------------------------------------------------------------------------

class PhaseEvaluator:
    """
    Evaluates the student model at configurable checkpoints during curriculum training.

    Supports two evaluation triggers:
      1. Phase transition  : evaluate when curriculum phase changes
      2. Periodic          : evaluate every `eval_every_n` iterations

    Results are stored in `history` and can be printed/logged at any time.

    Usage:
        phase_eval = PhaseEvaluator(
            evaluator=DetectionEvaluator(num_classes=3, ...),
            ir_val_loader=...,
            device=device,
            eval_every_n=500,
        )

        # Inside training loop:
        phase_eval.step(
            model=student,
            global_step=trainer.global_step,
            current_phase=current_phase,
        )
    """

    def __init__(
        self,
        evaluator:       DetectionEvaluator,
        ir_val_loader:   DataLoader,
        device:          torch.device,
        eval_every_n:    Optional[int] = None,   # None = only at phase transitions
        eval_on_phases:  Optional[List] = None,  # subset of Phase enum values
        log_fn: Optional[callable] = None,
        # RGB val — evaluated in Phase 1 only
        rgb_val_loader:  Optional[DataLoader] = None,
        # Visualization
        vis_dir:         Optional[str] = None,   # None = no visualization
        vis_every_n:     Optional[int] = None,   # None = only on eval trigger
        vis_num_samples: int = 8,
        vis_score_thresh: float = 0.3,
        class_names:     Optional[List[str]] = None,
        thresh_scheduler: Optional[AdaptiveThresholdScheduler] = None,
        phase3_end:      int = 0,   # curriculum.phase3_end — start of Phase 4 IR focus ramp
        # Teachers — when provided, visualization draws a side-by-side comparison
        # grid of student + teachers via visualize_compare_models. Otherwise only
        # the student is visualized.
        rgb_teacher:     Optional["torch.nn.Module"] = None,
        ir_teacher:      Optional["torch.nn.Module"] = None,
    ) -> None:
        self.evaluator      = evaluator
        self.ir_val_loader  = ir_val_loader
        self.rgb_val_loader = rgb_val_loader
        self.device         = device
        self.eval_every_n   = eval_every_n
        self.eval_on_phases = eval_on_phases    # None = all phases
        self.log_fn         = log_fn or logger.info

        # Separate evaluator instance for RGB val (same config as IR evaluator)
        self._rgb_evaluator = DetectionEvaluator(
            num_classes=evaluator.num_classes,
            class_names=evaluator.class_names,
            iou_thresholds=evaluator.iou_thresholds,
            interp=evaluator.interp,
        ) if rgb_val_loader is not None else None

        self.vis_dir           = vis_dir
        self.vis_every_n       = vis_every_n
        self.vis_num_samples   = vis_num_samples
        self.vis_score_thresh  = vis_score_thresh
        self.class_names       = class_names
        self.thresh_scheduler  = thresh_scheduler
        self.phase3_end        = phase3_end
        self.rgb_teacher       = rgb_teacher
        self.ir_teacher        = ir_teacher

        # Best checkpoint tracking — global and per-phase
        self.best_map50: float = -1.0
        self.on_new_best_fn = None          # set via .register_best_fn(fn)

        self.best_map50_per_phase: Dict[str, float] = {}
        self.on_new_phase_best_fn = None    # set via .register_phase_best_fn(fn)

        # State
        self._last_phase     = None
        self._last_eval_step = -1

        # Results history: List[{"step": int, "phase": str, "trigger": str, ...metrics}]
        self.history: List[Dict] = []

    def state_dict(self) -> Dict:
        """Serializable evaluator progress for resumable Kaggle runs."""
        return {
            "best_map50": self.best_map50,
            "best_map50_per_phase": self.best_map50_per_phase,
            "last_phase": self._last_phase.name if self._last_phase is not None else None,
            "last_eval_step": self._last_eval_step,
            "history": self.history,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.best_map50 = float(state.get("best_map50", -1.0))
        self.best_map50_per_phase = dict(state.get("best_map50_per_phase", {}))
        self._last_eval_step = int(state.get("last_eval_step", -1))
        self.history = list(state.get("history", []))

        last_phase_name = state.get("last_phase")
        self._last_phase = None
        if last_phase_name:
            try:
                from scheduler import Phase
                self._last_phase = Phase[last_phase_name]
            except (KeyError, ImportError):
                logger.warning(f"Could not restore evaluator last_phase={last_phase_name}")

    def register_best_fn(self, fn) -> None:
        """Register a callback called when a new global best mAP@0.5 is achieved."""
        self.on_new_best_fn = fn

    def register_phase_best_fn(self, fn) -> None:
        """Register a callback called when a new per-phase best mAP@0.5 is achieved.
        fn(results: Dict) — results includes global_step, phase, mAP@0.5, etc.
        """
        self.on_new_phase_best_fn = fn

    def step(
        self,
        model:        "torch.nn.Module",
        global_step:  int,
        current_phase,
    ) -> Optional[Dict]:
        """
        Call once per training iteration. Evaluates if triggered.

        Returns the result dict if evaluation ran, else None.
        """
        triggered, trigger_reason = self._should_evaluate(global_step, current_phase)
        self._last_phase = current_phase

        if triggered:
            return self.evaluate(model, global_step, current_phase, trigger_reason)

        # Visualization-only trigger (no full eval)
        if (
            self.vis_dir is not None
            and self.vis_every_n is not None
            and global_step > 0
            and global_step % self.vis_every_n == 0
        ):
            self._visualize(model, global_step, current_phase, trigger_reason="vis_periodic")

        return None

    def evaluate(
        self,
        model,
        global_step:    int,
        current_phase,
        trigger_reason: str = "manual",
    ) -> Dict:
        """
        Run evaluation on the IR val set and log results.
        """
        self.log_fn(
            f"[Eval] step={global_step}  phase={current_phase.name}  "
            f"trigger={trigger_reason}  running on {len(self.ir_val_loader.dataset)} IR val images ..."
        )

        model.eval()
        self.evaluator.reset()

        with torch.no_grad():
            for batch in self.ir_val_loader:
                images, targets = batch
                images  = images.to(self.device)
                preds   = model(images)
                targets = [{k: v for k, v in t.items()} for t in targets]
                self.evaluator.update(preds, targets)

        results = self.evaluator.compute()
        results.update({
            "global_step":    global_step,
            "phase":          current_phase.name,
            "trigger":        trigger_reason,
        })

        # RGB val metrics — Phase 1 periodic evals + Phase 1→2 transition
        if self._rgb_evaluator is not None and (
            current_phase.name == "PHASE1_RGB_WARMUP"
            or trigger_reason == "phase_end:PHASE1_RGB_WARMUP"
        ):
            rgb_results = self._eval_rgb_val(model)
            results.update(rgb_results)

        self.history.append(results)
        self._last_eval_step = global_step
        self._log_results(results)

        # Global best checkpoint
        map50 = results.get("mAP@0.5", 0.0)
        if map50 > self.best_map50:
            self.best_map50 = map50
            self.log_fn(f"[Eval] New global best mAP@0.5={map50:.4f} at step={global_step}")
            if self.on_new_best_fn is not None:
                self.on_new_best_fn(results)

        # Per-phase best checkpoint — Phase 1 uses RGB mAP (trained on RGB only),
        # all other phases use IR mAP (target domain metric).
        phase_key = current_phase.name
        if current_phase.name == "PHASE1_RGB_WARMUP" and "rgb_mAP@0.5" in results:
            phase_map50      = results["rgb_mAP@0.5"]
            phase_metric_key = "rgb_mAP@0.5"
        else:
            phase_map50      = map50
            phase_metric_key = "mAP@0.5"

        if phase_map50 > self.best_map50_per_phase.get(phase_key, -1.0):
            self.best_map50_per_phase[phase_key] = phase_map50
            self.log_fn(
                f"[Eval] New best for {phase_key}: {phase_metric_key}={phase_map50:.4f} at step={global_step}"
            )
            if self.on_new_phase_best_fn is not None:
                self.on_new_phase_best_fn(results)

        # Optional visualization
        if self.vis_dir is not None:
            self._visualize(model, global_step, current_phase, trigger_reason)

        model.train()
        return results

    def print_history(self) -> None:
        """Print a summary table of all evaluations."""
        if not self.history:
            logger.info("No evaluations recorded yet.")
            return

        # Header
        header = f"{'Step':>8}  {'Phase':<25}  {'Trigger':<18}  {'mAP@0.5':>8}"
        if "mAP@0.5:0.95" in self.history[0]:
            header += f"  {'mAP@0.5:0.95':>12}"
        logger.info("\n" + "=" * len(header))
        logger.info(header)
        logger.info("=" * len(header))

        for r in self.history:
            row = (
                f"{r['global_step']:>8}  "
                f"{r['phase']:<25}  "
                f"{r['trigger']:<18}  "
                f"{r.get('mAP@0.5', 0.0):>8.4f}"
            )
            if "mAP@0.5:0.95" in r:
                row += f"  {r['mAP@0.5:0.95']:>12.4f}"
            logger.info(row)

        logger.info("=" * len(header) + "\n")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _should_evaluate(self, global_step: int, current_phase) -> Tuple[bool, str]:
        # Phase transition trigger
        if self._last_phase is not None and current_phase != self._last_phase:
            if self.eval_on_phases is None or self._last_phase in self.eval_on_phases:
                return True, f"phase_end:{self._last_phase.name}"

        # Periodic trigger
        if self.eval_every_n is not None:
            if global_step > 0 and global_step % self.eval_every_n == 0:
                return True, f"periodic:{self.eval_every_n}"

        return False, ""

    @torch.no_grad()
    def _eval_rgb_val(self, model) -> Dict:
        """Compute mAP and supervised loss on the RGB validation set."""
        self.log_fn(
            f"[RGB Eval] running on {len(self.rgb_val_loader.dataset)} RGB val images ..."
        )

        # --- mAP ---
        self._rgb_evaluator.reset()
        model.eval()
        for images, targets in self.rgb_val_loader:
            images = images.to(self.device)
            preds  = model(images)
            self._rgb_evaluator.update(preds, [{k: v for k, v in t.items()} for t in targets])
        map_results = self._rgb_evaluator.compute()

        # --- Supervised val loss (model in train mode to get loss dict from FCOS) ---
        model.train()
        total_loss = 0.0
        n_batches  = 0
        for images, targets in self.rgb_val_loader:
            images  = images.to(self.device)
            targets = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()}
                for t in targets
            ]
            loss_dict   = model(images, targets)
            total_loss += sum(v.item() for v in loss_dict.values())
            n_batches  += 1
        model.eval()

        out = {f"rgb_{k}": v for k, v in map_results.items()}
        out["rgb_val_loss"] = total_loss / max(n_batches, 1)
        return out

    def _log_results(self, results: Dict) -> None:
        # --- IR val metrics ---
        map50 = results.get("mAP@0.5", 0.0)
        self.log_fn(f"[IR  Val] mAP@0.5={map50:.4f}")
        per_class = {k: v for k, v in results.items() if k.startswith("AP@")}
        for name, ap in sorted(per_class.items()):
            self.log_fn(f"  {name} = {ap:.4f}")
        if "mAP@0.5:0.95" in results:
            self.log_fn(f"  mAP@0.5:0.95 = {results['mAP@0.5:0.95']:.4f}")

        # --- RGB val metrics (Phase 1 only) ---
        if "rgb_mAP@0.5" in results:
            rgb_map50 = results["rgb_mAP@0.5"]
            rgb_loss  = results.get("rgb_val_loss", float("nan"))
            self.log_fn(f"[RGB Val] mAP@0.5={rgb_map50:.4f}  val_loss={rgb_loss:.4f}")
            rgb_per_class = {k: v for k, v in results.items() if k.startswith("rgb_AP@")}
            for name, ap in sorted(rgb_per_class.items()):
                self.log_fn(f"  {name} = {ap:.4f}")

    def _visualize(self, model, global_step: int, current_phase, trigger_reason: str) -> None:
        try:
            from visualize import visualize_compare_models, visualize_eval_samples
        except ImportError:
            self.log_fn("[Visualize] matplotlib not available, skipping visualization")
            return

        filename = f"step{global_step:07d}_{current_phase.name}_{trigger_reason}.png"
        save_path = str(Path(self.vis_dir) / filename)
        map50 = self.history[-1].get("mAP@0.5", 0.0) if self.history else 0.0
        title = (
            f"step={global_step}  phase={current_phase.name}  "
            f"trigger={trigger_reason}  mAP@0.5={map50:.4f}"
        )
        # Phase 4 ramp: steps_into_phase measured from phase3_end (start of Phase 4)
        steps_into_phase = (
            max(0, global_step - self.phase3_end)
            if current_phase.name == "PHASE4_IR_FOCUS"
            else 0
        )

        if self.thresh_scheduler is not None:
            rgb_thresh = self.thresh_scheduler.rgb_teacher(current_phase, steps_into_phase)
            ir_thresh  = self.thresh_scheduler.ir_teacher( current_phase, steps_into_phase)
        else:
            rgb_thresh = ir_thresh = self.vis_score_thresh

        # If teachers are registered, draw a side-by-side comparison grid.
        if self.rgb_teacher is not None or self.ir_teacher is not None:
            models: Dict[str, "torch.nn.Module"] = {"student": model}
            if self.rgb_teacher is not None:
                models["rgb_teacher"] = self.rgb_teacher
            if self.ir_teacher is not None:
                models["ir_teacher"] = self.ir_teacher

            # student → show all boxes (thresh=0.0)
            # teachers → per-class threshold from adaptive scheduler
            per_model_thresh: Dict[str, object] = {"student": 0.0}
            if self.rgb_teacher is not None:
                per_model_thresh["rgb_teacher"] = rgb_thresh
            if self.ir_teacher is not None:
                per_model_thresh["ir_teacher"] = ir_thresh

            visualize_compare_models(
                models=models,
                val_loader=self.ir_val_loader,
                device=self.device,
                save_path=save_path,
                num_samples=self.vis_num_samples,
                per_model_thresh=per_model_thresh,
                class_names=self.class_names,
                title=title,
            )
            return

        # No teachers — student only, show all boxes
        visualize_eval_samples(
            model=model,
            val_loader=self.ir_val_loader,
            device=self.device,
            save_path=save_path,
            num_samples=self.vis_num_samples,
            score_thresh=0.0,
            class_names=self.class_names,
            title=title,
        )
