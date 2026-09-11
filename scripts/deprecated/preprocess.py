"""
preprocess.py — preprocesa imágenes citogenéticas raw antes del pipeline.

uso:
    python scripts/preprocess.py \
        --input_dir data/raw/unmarked \
        --output_dir data/processed \
        --target_size 512 \
        --seed 42
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


# logging básico a fichero y consola
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
    # solo informativo, no es crítico aquí
    try:
        import torch
        if torch.cuda.is_available():
            logging.info(f"cuda: {torch.cuda.get_device_name(0)}")
        else:
            logging.warning("sin cuda, se usa cpu")
    except ImportError:
        logging.warning("pytorch no instalado")


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_image(path: Path) -> np.ndarray | None:
    # devuelve None si opencv no puede leerlo
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        logging.warning(f"no se pudo leer: {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_image(img: np.ndarray, target_size: int) -> np.ndarray:
    # lanczos da mejor calidad que bilineal para downscaling
    return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)


def remove_circular_artifacts(img: np.ndarray) -> np.ndarray:
    # detecta círculos con hough y los suaviza con apertura morfológica
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=50,
    )

    result = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        mask = np.zeros(gray.shape, dtype=np.uint8)
        for c in circles[0]:
            cv2.circle(mask, (c[0], c[1]), c[2], 255, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(
            result if len(result.shape) == 2 else cv2.cvtColor(result, cv2.COLOR_BGR2GRAY),
            cv2.MORPH_OPEN, kernel
        )
        if len(result.shape) == 2:
            result[mask == 255] = opened[mask == 255]
        else:
            result[:, :, 0][mask == 255] = opened[mask == 255]
        logging.debug(f"  círculos detectados: {circles.shape[1]}")

    # apertura global ligera para quitar ruido puntual
    kernel_global = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if len(result.shape) == 2:
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel_global)
    return result


def preprocess_image(
    img: np.ndarray,
    target_size: int | None,
    convert_gray: bool,
    remove_artifacts: bool,
) -> np.ndarray:
    if convert_gray:
        img = to_grayscale(img)
    if remove_artifacts:
        img = remove_circular_artifacts(img)
    if target_size is not None:
        img = resize_image(img, target_size)
    return img


def save_image(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="preprocesamiento de imágenes citogenéticas."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--target_size",
        default="512",
        help="resolución destino en píxeles o 'keep' para no redimensionar",
    )
    parser.add_argument("--convert_gray", action="store_true")
    parser.add_argument("--remove_artifacts", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--exp_name", default="preprocess")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_file = Path("results/logs") / f"{args.exp_name}.log"
    logger = setup_logging(log_file, args.log_level)
    fix_seed(args.seed)
    check_cuda()

    logger.info("=" * 60)
    logger.info("PREPROCESAMIENTO")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    target_size: int | None = None
    if args.target_size.lower() != "keep":
        target_size = int(args.target_size)

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS
    )
    if not image_paths:
        logger.error(f"no hay imágenes en {input_dir}")
        sys.exit(1)

    logger.info(f"imágenes encontradas: {len(image_paths)}")
    processed, skipped = 0, 0

    for img_path in tqdm(image_paths, desc="preprocesando", unit="img"):
        img = load_image(img_path)
        if img is None:
            skipped += 1
            continue
        try:
            img_out = preprocess_image(img, target_size, args.convert_gray, args.remove_artifacts)
            out_path = output_dir / img_path.name
            save_image(img_out, out_path)
            processed += 1
            logger.debug(f"  guardado: {out_path}")
        except Exception as e:
            logger.warning(f"error en {img_path.name}: {e}")
            skipped += 1

    logger.info(f"listo: {processed} procesadas, {skipped} omitidas")


if __name__ == "__main__":
    main()
