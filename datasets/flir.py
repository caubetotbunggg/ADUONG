"""
FLIR ADAS Aligned Dataset Loader — VOC XML format.

Actual dataset structure:
  align/
  ├── JPEGImages/
  │   ├── FLIR_XXXXX_PreviewData.jpeg   ← IR (thermal) image
  │   ├── FLIR_XXXXX_RGB.jpg            ← RGB image (spatially aligned with IR)
  │   └── ...
  ├── Annotations/
  │   ├── FLIR_XXXXX_PreviewData.xml    ← VOC XML annotation for IR image
  │   └── ...
  └── ImageSets/Main/
      ├── align_train.txt               ← stems: "FLIR_XXXXX_PreviewData"
      └── align_validation.txt

Key conventions:
  - stem  = "FLIR_XXXXX_PreviewData"
  - IR  image  : JPEGImages/{stem}.jpeg
  - RGB image  : JPEGImages/{stem.replace('_PreviewData','')}_RGB.jpg
  - Annotation : Annotations/{stem}.xml

Classes (FCOS 0-indexed, background excluded):
  person  → 0
  car     → 1
  bicycle → 2

Ignored labels: "FLIR" (source tag in XML), "dog" (too rare)

Domain adaptation roles:
  FLIRRGBDataset   → source domain  (labeled RGB, annotations from aligned IR XML)
  FLIRIRDataset    → target domain  (unlabeled IR, training only)
  FLIRIRValDataset → evaluation     (labeled IR, mAP computation)
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

FLIR_CLASSES    = ["person", "car", "bicycle"]
FLIR_CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(FLIR_CLASSES)}
NUM_CLASSES     = len(FLIR_CLASSES)   # 3

# COCO 91-class output indices for each FLIR class (used to slice COCO cls_logits).
# COCO label space: 0=__background__, 1=person, 2=bicycle, 3=car, ...
# FLIR order:        person(0)          car(1)              bicycle(2)
FLIR_TO_COCO_IDX: List[int] = [1, 3, 2]

# Labels to silently skip (noise / artefacts in the XML)
_IGNORE_LABELS  = {"FLIR", "dog"}


# ---------------------------------------------------------------------------
# VOC XML parser
# ---------------------------------------------------------------------------

def _parse_voc_xml(xml_path: Path) -> List[Dict]:
    """
    Parse a VOC-format XML file and return a list of object dicts.

    Each dict:  {"label": str,  "box": [xmin, ymin, xmax, ymax]}
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []

    root = tree.getroot()
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        if name in _IGNORE_LABELS or name not in FLIR_CLASS_TO_IDX:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        if xmax > xmin and ymax > ymin:
            objects.append({"label": name, "box": [xmin, ymin, xmax, ymax]})
    return objects


def _objects_to_tensors(
    objects: List[Dict],
    min_area: float = 16.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert parsed objects to (boxes [N,4], labels [N]) tensors."""
    boxes, labels = [], []
    for obj in objects:
        x1, y1, x2, y2 = obj["box"]
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(FLIR_CLASS_TO_IDX[obj["label"]])

    if boxes:
        return (
            torch.as_tensor(boxes,  dtype=torch.float32),
            torch.as_tensor(labels, dtype=torch.int64),
        )
    return (
        torch.zeros((0, 4), dtype=torch.float32),
        torch.zeros((0,),   dtype=torch.int64),
    )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _ir_stem_to_rgb_filename(stem: str) -> str:
    """FLIR_XXXXX_PreviewData  →  FLIR_XXXXX_RGB.jpg"""
    base = stem.replace("_PreviewData", "")
    return f"{base}_RGB.jpg"


def _read_split_file(split_file: Path) -> List[str]:
    """Read align_train.txt / align_validation.txt — one stem per line, ignoring blank lines."""
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Default transforms
# ---------------------------------------------------------------------------

def _default_rgb_transform() -> T.Compose:
    """ToTensor only — FCOS GeneralizedRCNNTransform handles ImageNet normalisation."""
    return T.Compose([T.ToTensor()])


def _default_ir_transform() -> T.Compose:
    """IR (thermal JPEG): convert to 3-channel float [0,1]. FCOS expects 3-ch input."""
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class FLIRRGBDataset(Dataset):
    """
    Labeled RGB source domain.

    Loads: JPEGImages/FLIR_XXXXX_RGB.jpg
    Labels: parsed from Annotations/FLIR_XXXXX_PreviewData.xml
    (annotations are aligned → valid for paired RGB images)

    Args:
        root       : path to the `align/` directory
        split      : "train" or "validation"
        transform  : image transform (default: ToTensor)
        min_area   : skip boxes smaller than this (pixels²)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        min_area: float = 16.0,
    ) -> None:
        self.root      = Path(root)
        self.transform = transform or _default_rgb_transform()
        self.min_area  = min_area

        split_file = self.root / f"align_{split}.txt"
        self.stems = _read_split_file(split_file)

        # Filter to stems where the RGB image actually exists
        self.stems = [
            s for s in self.stems
            if (self.root / "JPEGImages" / _ir_stem_to_rgb_filename(s)).exists()
        ]

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        stem     = self.stems[idx]
        rgb_path = self.root / "JPEGImages" / _ir_stem_to_rgb_filename(stem)
        ann_path = self.root / "Annotations" / f"{stem}.xml"

        img    = Image.open(rgb_path).convert("RGB")
        img_t  = self.transform(img)

        objects       = _parse_voc_xml(ann_path)
        boxes, labels = _objects_to_tensors(objects, self.min_area)

        target = {"boxes": boxes, "labels": labels, "stem": stem}
        return img_t, target


class FLIRIRDataset(Dataset):
    """
    Unlabeled IR target domain — for training only.

    Loads: JPEGImages/FLIR_XXXXX_PreviewData.jpeg  (no labels)

    Args:
        root      : path to the `align/` directory
        split     : "train" or "validation"
        transform : image transform (default: Grayscale→3ch, ToTensor)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        self.root      = Path(root)
        self.transform = transform or _default_ir_transform()

        split_file = self.root / f"align_{split}.txt"
        all_stems  = _read_split_file(split_file)

        self.ir_paths = [
            self.root / "JPEGImages" / f"{s}.jpeg"
            for s in all_stems
            if (self.root / "JPEGImages" / f"{s}.jpeg").exists()
        ]

    def __len__(self) -> int:
        return len(self.ir_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        img   = Image.open(self.ir_paths[idx])
        img_t = self.transform(img)
        return (img_t,)


class FLIRIRValDataset(Dataset):
    """
    Labeled IR target domain — for evaluation only.

    Loads: JPEGImages/FLIR_XXXXX_PreviewData.jpeg
    Labels: Annotations/FLIR_XXXXX_PreviewData.xml

    Args:
        root      : path to the `align/` directory
        split     : "train" or "validation"
        transform : image transform
        min_area  : skip boxes smaller than this (pixels²)
    """

    def __init__(
        self,
        root: str,
        split: str = "validation",
        transform: Optional[Callable] = None,
        min_area: float = 16.0,
    ) -> None:
        self.root      = Path(root)
        self.transform = transform or _default_ir_transform()
        self.min_area  = min_area

        split_file = self.root / f"align_{split}.txt"
        all_stems  = _read_split_file(split_file)

        # Keep only stems where both image and annotation exist
        self.stems = [
            s for s in all_stems
            if (self.root / "JPEGImages"  / f"{s}.jpeg").exists()
            and (self.root / "Annotations" / f"{s}.xml").exists()
        ]

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        stem     = self.stems[idx]
        ir_path  = self.root / "JPEGImages"  / f"{stem}.jpeg"
        ann_path = self.root / "Annotations" / f"{stem}.xml"

        img    = Image.open(ir_path)
        img_t  = self.transform(img)

        objects       = _parse_voc_xml(ann_path)
        boxes, labels = _objects_to_tensors(objects, self.min_area)

        target = {"boxes": boxes, "labels": labels, "stem": stem}
        return img_t, target


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def rgb_collate(batch: List) -> Tuple[torch.Tensor, List[Dict]]:
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def ir_collate(batch: List) -> Tuple[torch.Tensor]:
    return (torch.stack([b[0] for b in batch]),)


def ir_val_collate(batch: List) -> Tuple[torch.Tensor, List[Dict]]:
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]
