"""
01_augment.py — Data augmentation consistente (imagen + bounding boxes YOLO).

Librería: Albumentations (soporta bbox transforms nativo).

Uso:
    python scripts/01_augment.py \
        --input_dir data/processed \
        --annot_dir data/annotations \
        --output_dir data/augmented/images \
        --output_annot data/augmented/annotations \
        --factor 3 \
        --seed 42
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    import albumentations as A
    from albumentations.core.composition import Compose
except ImportError:
    print("ERROR: albumentations no está instalado. Ejecuta: pip install albumentations")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path, level: str = "INFO") -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def check_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            logging.info(f"CUDA disponible: {torch.cuda.get_device_name(0)}")
        else:
            logging.warning("CUDA NO disponible.")
    except ImportError:
        pass


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Formato YOLO  →  [class_id, cx, cy, w, h]  (normalizadas [0,1])
# ---------------------------------------------------------------------------

def load_yolo_annotations(annot_path: Path) -> list[list[float]]:
    """Devuelve lista de [class_id, cx, cy, w, h]."""
    if not annot_path.exists():
        return []
    boxes = []
    for line in annot_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            boxes.append([float(p) for p in parts])
    return boxes


def save_yolo_annotations(boxes: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for b in boxes:
        class_id = int(b[0])
        lines.append(f"{class_id} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}")
    path.write_text("\n".join(lines))


def yolo_to_albumentation(boxes: list) -> tuple[list, list]:
    """Convierte YOLO [cls, cx, cy, w, h] → albumentations [x_min, y_min, x_max, y_max]."""
    alb_boxes, class_ids = [], []
    for b in boxes:
        cls, cx, cy, bw, bh = b
        x_min = max(0.0, cx - bw / 2)
        y_min = max(0.0, cy - bh / 2)
        x_max = min(1.0, cx + bw / 2)
        y_max = min(1.0, cy + bh / 2)
        alb_boxes.append([x_min, y_min, x_max, y_max])
        class_ids.append(int(cls))
    return alb_boxes, class_ids


def albumentation_to_yolo(alb_boxes: list, class_ids: list) -> list:
    """Convierte albumentations → YOLO."""
    yolo_boxes = []
    for (x_min, y_min, x_max, y_max), cls in zip(alb_boxes, class_ids):
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        bw = x_max - x_min
        bh = y_max - y_min
        yolo_boxes.append([cls, cx, cy, bw, bh])
    return yolo_boxes


# ---------------------------------------------------------------------------
# Pipeline de augmentación
# ---------------------------------------------------------------------------

def build_transform(seed: int) -> Compose:
    return A.Compose(
        [
            # Geométricas (reversibles con bbox)
            A.Rotate(limit=15, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomScale(scale_limit=0.15, p=0.5),           # zoom ±15%
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0, rotate_limit=0, p=0.4),
            # Fotométricas (no afectan bboxes)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.GaussNoise(var_limit=(0, 25), p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
        ],
        bbox_params=A.BboxParams(
            format="albumentations",
            label_fields=["class_labels"],
            min_area=0.0,
            min_visibility=0.1,
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data augmentation consistente para imágenes citogenéticas + anotaciones YOLO."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--annot_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_annot", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=3, choices=[2, 3, 4, 5],
                        help="Multiplicador del dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--exp_name", default="augment")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    log_file = Path("results/logs") / f"{args.exp_name}.log"
    setup_logging(log_file, args.log_level)
    fix_seed(args.seed)
    check_cuda()

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("SCRIPT 01 — DATA AUGMENTATION")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_annot.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in args.input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS
    )
    if not image_paths:
        logger.error(f"No se encontraron imágenes en {args.input_dir}.")
        sys.exit(1)

    logger.info(f"Imágenes originales: {len(image_paths)} | Factor: x{args.factor}")
    total_aug = 0

    for img_path in tqdm(image_paths, desc="Augmentando", unit="img"):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning(f"No se pudo leer: {img_path}. Se omite.")
            continue

        # Convertir a RGB si es necesario para Albumentations
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img

        annot_path = args.annot_dir / img_path.with_suffix(".txt").name
        yolo_boxes = load_yolo_annotations(annot_path)
        alb_boxes, class_ids = yolo_to_albumentation(yolo_boxes)

        for aug_idx in range(1, args.factor):
            transform = build_transform(args.seed + aug_idx * 137)
            try:
                result = transform(image=img_rgb, bboxes=alb_boxes, class_labels=class_ids)
            except Exception as e:
                logger.warning(f"Error augmentando {img_path.name} aug{aug_idx}: {e}")
                continue

            aug_img = result["image"]
            aug_boxes_alb = result["bboxes"]
            aug_class_ids = result["class_labels"]
            aug_boxes_yolo = albumentation_to_yolo(aug_boxes_alb, aug_class_ids)

            stem = img_path.stem
            ext = img_path.suffix
            out_img_path = args.output_dir / f"{stem}_aug{aug_idx}{ext}"
            out_ann_path = args.output_annot / f"{stem}_aug{aug_idx}.txt"

            # Guardar imagen
            if len(img.shape) == 2:
                save_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2GRAY)
            else:
                save_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_img_path), save_img)
            save_yolo_annotations(aug_boxes_yolo, out_ann_path)
            total_aug += 1

    logger.info(f"Augmentación completa: {total_aug} imágenes nuevas generadas.")


if __name__ == "__main__":
    main()
