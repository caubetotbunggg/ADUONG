# DAOD-RGB2IR

**Curriculum Domain Adaptation for RGB-to-IR Object Detection**

Framework huấn luyện chuyển giao miền (Domain Adaptation) từ ảnh RGB sang ảnh hồng ngoại (IR / Thermal) cho bài toán phát hiện đối tượng (Object Detection), sử dụng chiến lược Curriculum Learning 4 pha kết hợp với SAGA (Semantic-Aware Grayscale Augmentation) và AEMA (Adaptive EMA).

---

## 1. Tổng quan Framework

Dự án giải quyết bài toán **Unsupervised Domain Adaptation (UDA)** trong Object Detection:
- **Source domain**: RGB có nhãn (labeled)
- **Target domain**: IR (thermal) không nhãn (unlabeled)
- **Mục tiêu**: Huấn luyện model phát hiện đối tượng hoạt động tốt trên ảnh IR mà không cần nhãn IR.

Framework sử dụng kiến trúc **1 Student + 2 Teachers**:
- **Student**: Model chính được huấn luyện bằng gradient descent.
- **RGB Teacher**: EMA teacher chuyên biệt hóa cho miền RGB, cập nhật trong Phase 2.
- **IR Teacher**: EMA teacher chuyên biệt hóa cho miền IR, cập nhật trong Phase 3 và 4.

---

## 2. Kiến trúc Tổng thể

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Curriculum Domain Adaptation Trainer                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   RGB DataLoader          IR DataLoader                                     │
│        │                       │                                            │
│        ▼                       ▼                                            │
│   ┌─────────┐            ┌─────────┐                                        │
│   │  SAGA   │◄───────────│  MID   │──► Intermediate Domain Bridge         │
│   └────┬────┘            └─────────┘                                        │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                     4-PHASE CURRICULUM                          │      │
│   │  Phase 1: RGB only        (supervised warmup)                   │      │
│   │  Phase 2: RGB + MID       (mixed-batch domain bridge)           │      │
│   │  Phase 3: MID + IR        (mixed-batch domain bridge)           │      │
│   │  Phase 4: IR only         (unsupervised IR focus)               │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │              1 STUDENT  +  2 TEACHERS (EMA/AEMA)                │      │
│   │                                                                 │      │
│   │   Student ──► rgb_teacher (EMA/AEMA)  ──► Pseudo-labels RGB    │      │
│   │         └──► ir_teacher  (EMA/AEMA)  ──► Pseudo-labels IR     │      │
│   │                                                                 │      │
│   │   Loss = GT_loss + λ_rgb·Pseudo_rgb + λ_ir·Pseudo_ir           │      │
│   │          + λ_adv·Adversarial_alignment (Phase 2/3)             │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │              Adaptive Threshold + Phase Evaluator               │      │
│   │   • Per-class confidence thresholds                             │      │
│   │   • Phase 4 linear ramp-up                                      │      │
│   │   • mAP@0.5 / mAP@0.5:0.95 evaluation                           │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Các Thành phần Chính

### 3.1. SAGA — Semantic-Aware Grayscale Augmentation (`saga.py`)

SAGA là **cầu nối miền** (domain bridge) giữa RGB và IR:
- Chỉ chuyển vùng **đối tượng** (bên trong bounding boxes) sang grayscale.
- Giữ nguyên **background** ở dạng RGB.
- Tạo ra miền trung gian **MID** (Mixed Intermediate Domain) nằm giữa RGB và IR.

```
RGB (color + texture)  ──►  MID/SAGA (gray objects, RGB background)  ──►  IR (grayscale)
```

Có hai biến thể:
- **Hard SAGA**: Chuyển hoàn toàn vùng object sang grayscale (binary mask).
- **Soft SAGA**: Blend giữa RGB và grayscale với alpha (`pixel_out = alpha * rgb + (1-alpha) * gray`).

### 3.2. AEMA — Adaptive EMA (`ema.py`)

AEMA (DDT-style Adaptive EMA) là cơ chế cập nhật teacher thông minh:
- Tích lũy gradient magnitude của teacher theo thời gian.
- **Top 10%** parameters có gradient lớn nhất được cập nhật với `fast_alpha = 0.997`.
- **90% còn lại** cập nhật với `slow_alpha = 0.9996`.
- Giúp teacher thích ứng nhanh với những vùng quan trọng trong feature space.

Ngoài ra cũng hỗ trợ **Classic EMA** với warmup schedule.

### 3.3. Luồng Huấn luyện 4 Pha (`trainer.py`, `scheduler.py`)

| Phase | Tên | Batch | GT Labels | Teachers | EMA Update |
|-------|-----|-------|-----------|----------|------------|
| **1** | RGB Warmup | RGB | Có | Không | Không |
| **2** | RGB + MID | Mixed `[RGB \| MID]` | Cả batch | rgb_teacher + ir_teacher | rgb_teacher only |
| **3** | MID + IR | Mixed `[MID \| IR]` | Chỉ MID slice | rgb_teacher + ir_teacher | ir_teacher only |
| **4** | IR Focus | IR | Không | ir_teacher | ir_teacher only |

**Mixed-batch mechanism**:
- Phase 2: Batch được chia đôi — một nửa RGB gốc, một nửa SAGA-transformed RGB.
- Phase 3: Batch gồm SAGA-transformed RGB (có GT) và ảnh IR thực (không GT).

### 3.4. Adversarial Domain Alignment (`discriminator.py`)

Sử dụng **DANN-style** Gradient Reversal Layer (GRL):
- **Phase 2**: Discriminator `disc_rgb` phân biệt RGB (label=0) vs MID (label=1).
- **Phase 3**: Discriminator `disc_ir` phân biệt MID (label=0) vs IR (label=1).
- GRL lambda tăng dần theo schedule để ổn định huấn luyện.

### 3.5. Adaptive Threshold (`adaptive_threshold.py`)

Confidence threshold cho pseudo-labels thay đổi theo phase:
- **Per-class thresholds**: person, car, bicycle có ngưỡng khác nhau.
- **Phase 4 ramp-up**: Threshold tăng tuyến tính từ start → end để giảm False Positives khi model đã ổn định.

### 3.6. Evaluator (`evaluator.py`)

Tính mAP **không phụ thuộc pycocotools**:
- VOC 11-point interpolation AP.
- COCO-style mAP@0.5:0.95 (AUC interpolation).
- Per-class AP tracking.
- Tự động đánh giá khi chuyển phase và định kỳ.

---

## 4. Cấu trúc Thư mục

```
DAOD-RGB2IR/
├── config.py                  # Dataclass configs (EMA, SAGA, Curriculum, Loss, ...)
├── saga.py                    # Semantic-Aware Grayscale Augmentation
├── trainer.py                 # CurriculumDomainAdaptationTrainer (4-phase loop)
├── scheduler.py               # CurriculumScheduler (phase/step dispatch)
├── losses.py                  # Loss functions cho từng phase
├── ema.py                     # Classic EMA + AEMA updater
├── discriminator.py           # DomainDiscriminator + GRL + DANN schedule
├── adaptive_threshold.py      # Per-class adaptive confidence thresholds
├── evaluator.py               # DetectionEvaluator + PhaseEvaluator (mAP)
├── batch_types.py             # Typed batch containers (RGBBatch, IRBatch, ...)
├── visualize.py               # Visualization utilities
├── faster_rcnn_wrapper.py     # Faster R-CNN builder (student + 2 teachers)
├── fcos_wrapper.py            # FCOS builder (student + 2 teachers)
├── example_flir.py            # Entry point script cho FLIR ADAS Aligned dataset
│
├── datasets/
│   └── flir.py               # FLIRRGBDataset, FLIRIRDataset, FLIRIRValDataset
│
├── scripts/
│   └── kaggle_train_aema.sh  # Script chạy trên Kaggle
│
├── tests/
│   └── test_ema.py           # Unit tests cho EMA
│
├── requirements.txt
└── README.md
```

---

## 5. Cài đặt

### Yêu cầu

- Python ≥ 3.10
- PyTorch ≥ 2.0.0
- torchvision ≥ 0.15.0
- Pillow ≥ 9.0.0
- matplotlib ≥ 3.5.0

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

---

## 6. Chuẩn bị Dữ liệu

Framework được thiết kế cho **FLIR ADAS Aligned Dataset**.

### Cấu trúc thư mục dữ liệu

```
align/
├── JPEGImages/
│   ├── FLIR_XXXXX_PreviewData.jpeg   # Ảnh IR (thermal)
│   ├── FLIR_XXXXX_RGB.jpg            # Ảnh RGB (aligned với IR)
│   └── ...
├── Annotations/
│   ├── FLIR_XXXXX_PreviewData.xml    # VOC XML annotations (cho IR)
│   └── ...
└── ImageSets/Main/
    ├── align_train.txt               # 4129 training stems
    └── align_validation.txt          # 1013 validation stems
```

### Các lớp đối tượng (FCOS 0-indexed)

| Index | Class   |
|-------|---------|
| 0     | person  |
| 1     | car     |
| 2     | bicycle |

**Lưu ý**: Annotation được lấy từ XML của IR image, nhưng vì RGB và IR đã được spatially aligned nên cùng một annotation áp dụng cho cả hai.

---

## 7. Huấn luyện

### Huấn luyện cơ bản (FCOS)

```bash
python example_flir.py \
  --data_root /path/to/align \
  --device cuda \
  --output_dir ./output \
  --total_iters 35000 \
  --batch_size 4
```

### Huấn luyện trên Apple Silicon (MPS)

```bash
python example_flir.py \
  --data_root /path/to/align \
  --device mps \
  --output_dir ./output
```

### Smoke test nhanh (không cần pretrained backbone)

```bash
python example_flir.py \
  --data_root /path/to/align \
  --device mps \
  --no_pretrained_backbone \
  --total_iters 1 \
  --batch_size 1 \
  --workers 0 \
  --eval_every 999999 \
  --vis_every 999999
```

---

## 8. Tham số CLI Quan trọng

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--data_root` | `None` | Đường dẫn tới thư mục `align/` |
| `--output_dir` | `./output` | Thư mục lưu checkpoint và metrics |
| `--total_iters` | `35000` | Tổng số iteration huấn luyện |
| `--batch_size` | `4` | Batch size |
| `--lr_backbone` | `5e-5` | Learning rate cho backbone |
| `--lr_head` | `5e-4` | Learning rate cho detection head |
| `--device` | `cuda` | Thiết bị: `cuda`, `mps`, `cpu` |
| `--model` | `fcos` | Detector: `fcos` hoặc `faster_rcnn` |

### Tham số Curriculum

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--phase1_end` | `15000` | Kết thúc Phase 1 (RGB warmup) |
| `--phase2_end` | `20000` | Kết thúc Phase 2 (RGB + MID) |
| `--phase3_end` | `25000` | Kết thúc Phase 3 (MID + IR) |

### Tham số EMA/AEMA

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--ema_mode` | `aema` | `aema` (adaptive) hoặc `ema` (classic) |
| `--ema_alpha` | `0.9996` | Classic EMA decay |
| `--aema_fast_alpha` | `0.997` | AEMA fast update rate (top 10%) |
| `--aema_slow_alpha` | `0.9996` | AEMA slow update rate (rest 90%) |
| `--aema_top_ratio` | `0.10` | Phần trăm params dùng fast alpha |

### Tham số Adversarial

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--adv_weight` | `0.2` | Trọng số adversarial loss (0 = tắt) |
| `--disc_lr` | `1e-4` | Learning rate cho discriminator |
| `--grl_lambda` | `1.0` | Max GRL lambda |
| `--no_grl_schedule` | `False` | Dùng fixed lambda thay vì DANN schedule |

### Tham số Đánh giá & Checkpoint

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--eval_every` | `2000` | Đánh giá mỗi N iterations |
| `--vis_every` | `500` | Visualize mỗi N iterations |
| `--save_every` | `5000` | Lưu checkpoint mỗi N iterations |
| `--skip_eval` | `False` | Bỏ qua evaluation (cho smoke test) |

---

## 9. Resume Training

### Tự động resume từ checkpoint gần nhất

```bash
python example_flir.py \
  --data_root /path/to/align \
  --device cuda \
  --auto_resume
```

### Resume từ checkpoint cụ thể

```bash
python example_flir.py \
  --data_root /path/to/align \
  --device cuda \
  --resume ./output/ckpt_015000.pt
```

---

## 10. Kiến trúc Loss theo Phase

### Phase 1 — RGB Warmup

```
Loss = p1_gt_weight · L_gt(RGB)
```

### Phase 2 — Mixed [RGB | MID]

```
Loss = p2_gt_weight          · L_gt(whole batch)
     + p2_rgb_teacher_weight · L_pseudo(rgb_teacher, whole batch)
     + p2_ir_teacher_weight  · L_pseudo(ir_teacher,  whole batch)
     + p2_adv_weight         · L_adversarial(disc_rgb)
```

### Phase 3 — Mixed [MID | IR]

```
Loss = p3_gt_weight          · L_gt(MID slice only)
     + p3_rgb_teacher_weight · L_pseudo(rgb_teacher, whole batch)
     + p3_ir_teacher_weight  · L_pseudo(ir_teacher,  whole batch)
     + p3_adv_weight         · L_adversarial(disc_ir)
```

### Phase 4 — IR Focus

```
Loss = p4_ir_teacher_weight · L_pseudo(ir_teacher, IR)
```

---

## 11. Cấu hình Mặc định (config.py)

```python
TrainingConfig(
    ema=EMAConfig(mode="aema", alpha=0.999, ...),
    saga=SAGAConfig(apply_prob=1.0),  # SAGA luôn được áp dụng
    rgb_aug=RGBAugConfig(hflip_prob=0.5, multiscale_min=0.5, ...),
    ir_aug=IRAugConfig(hflip_prob=0.5, intensity_shift_prob=0.5, ...),
    curriculum=CurriculumConfig(
        phase1_end=3000,
        phase2_end=10000,
        phase3_end=17000,
        phase2_rgb_ratio=0.5,
        phase3_mid_ratio=0.5,
    ),
    loss=LossConfig(
        p1_gt_weight=1.0,
        p2_gt_weight=1.0, p2_rgb_teacher_weight=0.5, p2_ir_teacher_weight=0.5,
        p3_gt_weight=1.0, p3_rgb_teacher_weight=0.5, p3_ir_teacher_weight=0.5,
        p4_ir_teacher_weight=1.0,
    ),
    adv=AdvConfig(p2_adv_weight=0.2, p3_adv_weight=0.2, ...),
    pseudo_label_conf_thresh=0.7,
    grad_clip=10.0,
)
```

---

## 12. Hỗ trợ Detectors

| Detector | File | Đặc điểm |
|----------|------|----------|
| **FCOS** | `fcos_wrapper.py` | One-stage, anchor-free, mặc định |
| **Faster R-CNN** | `faster_rcnn_wrapper.py` | Two-stage, hỗ trợ focal loss gamma |

Cả hai đều dùng **ResNet50-FPN** backbone với ImageNet pretraining.

---

## 13. License

Apache License 2.0 — xem file [LICENSE](LICENSE).

---

## 14. Tài liệu Tham khảo

Framework này tổng hợp các kỹ thuật từ:
- **Curriculum Learning**: Tăng dần độ khó từ RGB → MID → IR
- **Mean Teacher / EMA**: Consistency regularization với teacher models
- **DDT (Dual-Domain Teacher)**: AEMA adaptive teacher update
- **DANN**: Gradient Reversal Layer cho adversarial domain alignment
- **SAGA**: Semantic-aware augmentation tạo miền trung gian
