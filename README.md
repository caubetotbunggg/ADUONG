# DAOD-RGB2IR

Domain Adaptive Object Detection từ RGB sang IR (thermal) sử dụng curriculum learning và GAN-based image translation.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc dataset (FLIR ADAS Aligned)

```
align/
├── JPEGImages/
│   ├── FLIR_XXXXX_PreviewData.jpeg   # IR (thermal)
│   ├── FLIR_XXXXX_RGB.jpg            # RGB (paired)
│   └── ...
├── Annotations/
│   ├── FLIR_XXXXX_PreviewData.xml    # VOC XML (cho IR)
│   └── ...
└── ImageSets/Main/
    ├── align_train.txt               # 4129 stems
    └── align_validation.txt          # 1013 stems
```

Classes: `person` (0), `car` (1), `bicycle` (2)

## Chạy training

Từ trong thư mục `DAOD-RGB2IR/`:

```bash
python example_flir.py --data_root /path/to/align --device cuda --from_coco
```

### Các tham số thường dùng

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--data_root` | *(bắt buộc)* | Đường dẫn tới thư mục `align/` |
| `--output_dir` | `./output` | Nơi lưu checkpoint và log |
| `--model` | `fcos` | Model backbone (`fcos` hoặc `faster_rcnn`) |
| `--total_iters` | 30000 | Số iteration training |
| `--batch_size` | 32 | Batch size |
| `--device` | `cuda` | `cuda` hoặc `cpu` |
| `--resume` | — | Tiếp tục từ checkpoint |
| `--gan_checkpoint` | — | Load GAN translator đã train sẵn |

### Tiếp tục từ checkpoint

```bash
python example_flir.py --data_root /path/to/align --resume ./output/checkpoint_10000.pth
```
