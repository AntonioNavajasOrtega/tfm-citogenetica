"""
degrade.py — genera imágenes lr simuladas a partir de las hr procesadas.

modos:
  - bicubic    : downsampling bicúbico clásico
  - bilinear   : downsampling bilineal
  - blur_noise : blur + ruido gaussiano + downsampling (simula condiciones reales)

uso:
    python scripts/degrade.py \
        --input_dir data/processed \
        --output_dir data/lr/x2 \
        --scale 2 \
        --degradation blur_noise
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


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
            logging.info(f"cuda: {torch.cuda.get_device_name(0)}")
        else:
            logging.warning("sin cuda")
    except ImportError:
        pass


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# mapa de interpolación por modo de degradación
INTERP_MAP = {
    "bicubic":    cv2.INTER_CUBIC,
    "bilinear":   cv2.INTER_LINEAR,
    "blur_noise": cv2.INTER_CUBIC,
}


def apply_blur_noise(img: np.ndarray, noise_sigma: float, blur_kernel: int) -> np.ndarray:
    # primero blur, luego ruido aditivo gaussiano
    if blur_kernel > 0 and blur_kernel % 2 == 1:
        img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def degrade_image(
    img: np.ndarray,
    scale: int,
    degradation: str,
    noise_sigma: float,
    blur_kernel: int,
) -> np.ndarray:
    h, w = img.shape[:2]
    lr_h, lr_w = h // scale, w // scale

    if degradation == "blur_noise":
        img = apply_blur_noise(img, noise_sigma, blur_kernel)

    interp = INTERP_MAP.get(degradation, cv2.INTER_CUBIC)
    lr = cv2.resize(img, (lr_w, lr_h), interpolation=interp)
    return lr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="genera imágenes lr simuladas para entrenamiento sr."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument(
        "--degradation",
        default="bicubic",
        choices=["bicubic", "bilinear", "blur_noise"],
    )
    parser.add_argument("--noise_sigma", type=float, default=5.0)
    parser.add_argument("--blur_kernel", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--exp_name", default="degrade")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_file = Path("results/logs") / f"{args.exp_name}.log"
    setup_logging(log_file, args.log_level)
    fix_seed(args.seed)
    check_cuda()

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("DEGRADACIÓN LR")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in args.input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS
    )
    if not image_paths:
        logger.error(f"no hay imágenes en {args.input_dir}")
        sys.exit(1)

    logger.info(f"imágenes hr: {len(image_paths)} | escala: x{args.scale} | modo: {args.degradation}")
    processed, skipped = 0, 0

    for img_path in tqdm(image_paths, desc="degradando", unit="img"):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning(f"no se pudo leer: {img_path}")
            skipped += 1
            continue
        try:
            lr = degrade_image(img, args.scale, args.degradation, args.noise_sigma, args.blur_kernel)
            # el nombre incluye el factor de escala para identificarlo después
            out_path = args.output_dir / f"{img_path.stem}_x{args.scale}{img_path.suffix}"
            cv2.imwrite(str(out_path), lr)
            processed += 1
            logger.debug(f"  guardado: {out_path} ({lr.shape[1]}x{lr.shape[0]})")
        except Exception as e:
            logger.warning(f"error en {img_path.name}: {e}")
            skipped += 1

    logger.info(f"listo: {processed} degradadas, {skipped} omitidas")


if __name__ == "__main__":
    main()
