"""
degrade.py — genera imágenes LR simuladas a partir de las imágenes raw originales.

En una sola ejecución produce los dos factores de escala (x2 y x4) bajo
la carpeta de salida, con la siguiente estructura:

    <output_dir>/
        x2/   ← imágenes degradadas a factor ×2
        x4/   ← imágenes degradadas a factor ×4

Modos de degradación disponibles:
  - blur_noise  (recomendado para microscopía)
                Simula la degradación real: desenfoque óptico (GaussianBlur)
                + ruido del sensor (ruido gaussiano aditivo) + downsampling
                bicúbico. Produce pares LR/HR más realistas que el simple
                downsampling.
  - bicubic     Downsampling bicúbico clásico, sin preprocesado extra.
  - bilinear    Downsampling bilineal.

Uso básico (genera x2 y x4 con blur_noise):
    python scripts/degrade.py --input_dir data/raw/unmarked

Con opciones:
    python scripts/degrade.py \
        --input_dir   data/raw/unmarked \
        --output_dir  data/lr \
        --degradation blur_noise \
        --noise_sigma 5.0 \
        --blur_kernel 3
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SCALES = [2, 4]

INTERP_MAP = {
    "bicubic":    cv2.INTER_CUBIC,
    "bilinear":   cv2.INTER_LINEAR,
    "blur_noise": cv2.INTER_CUBIC,   # downsampling final con bicúbico
}


# ---------------------------------------------------------------------------
# Logging y utilidades
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path, level: str = "INFO") -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Degradación
# ---------------------------------------------------------------------------

def apply_blur_noise(
    img: np.ndarray,
    noise_sigma: float,
    blur_kernel: int,
) -> np.ndarray:
    """
    Simula la degradación real de un microscopio:
      1. GaussianBlur  → desenfoque óptico / de movimiento
      2. Ruido aditivo → ruido del sensor (electrónico + fotónico)
    """
    # 1. Desenfoque
    if blur_kernel > 0 and blur_kernel % 2 == 1:
        img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)

    # 2. Ruido gaussiano aditivo
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
    """Aplica la degradación y hace downsampling al factor indicado."""
    h, w = img.shape[:2]
    lr_h, lr_w = h // scale, w // scale

    if degradation == "blur_noise":
        # El blur_kernel se escala con el factor para que el desenfoque sea
        # proporcional: x4 necesita más blur que x2.
        scaled_kernel = blur_kernel + (scale - 2)        # 3→3 para x2, 3→5 para x4
        if scaled_kernel % 2 == 0:
            scaled_kernel += 1                            # siempre impar
        img = apply_blur_noise(img, noise_sigma, scaled_kernel)

    interp = INTERP_MAP.get(degradation, cv2.INTER_CUBIC)
    lr = cv2.resize(img, (lr_w, lr_h), interpolation=interp)
    return lr


# ---------------------------------------------------------------------------
# Procesado de un factor de escala
# ---------------------------------------------------------------------------

def process_scale(
    image_paths: list[Path],
    scale: int,
    out_dir: Path,
    degradation: str,
    noise_sigma: float,
    blur_kernel: int,
    logger: logging.Logger,
) -> tuple[int, int]:
    """
    Degrada todas las imágenes al factor `scale` y las guarda en `out_dir`.
    Devuelve (procesadas, omitidas).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    processed, skipped = 0, 0

    for img_path in tqdm(image_paths, desc=f"  x{scale}", unit="img", leave=False):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning(f"    [!] No se pudo leer: {img_path.name}")
            skipped += 1
            continue
        try:
            lr = degrade_image(img, scale, degradation, noise_sigma, blur_kernel)
            # Conserva el nombre original; la carpeta ya indica el factor
            out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), lr)
            h_hr, w_hr = img.shape[:2]
            h_lr, w_lr = lr.shape[:2]
            logger.debug(
                f"    {img_path.name}: {w_hr}×{h_hr} → {w_lr}×{h_lr}"
            )
            processed += 1
        except Exception as e:
            logger.warning(f"    [!] Error en {img_path.name}: {e}")
            skipped += 1

    return processed, skipped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genera imágenes LR (×2 y ×4) a partir de imágenes raw."
    )
    p.add_argument(
        "--input_dir", type=Path, default=Path("data/raw/unmarked"),
        help="Carpeta con imágenes HR originales (default: data/raw/unmarked)"
    )
    p.add_argument(
        "--output_dir", type=Path, default=Path("data/lr"),
        help="Carpeta raíz de salida. Se crearán subcarpetas x2/ y x4/ "
             "(default: data/lr)"
    )
    p.add_argument(
        "--degradation", default="blur_noise",
        choices=["blur_noise", "bicubic", "bilinear"],
        help="Tipo de degradación (default: blur_noise — recomendado para microscopía)"
    )
    p.add_argument(
        "--noise_sigma", type=float, default=5.0,
        help="Desviación estándar del ruido gaussiano aditivo, solo blur_noise "
             "(default: 5.0)"
    )
    p.add_argument(
        "--blur_kernel", type=int, default=3,
        help="Tamaño del kernel GaussianBlur (impar), solo blur_noise "
             "(default: 3 para x2; se escala a 5 automáticamente para x4)"
    )
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    p.add_argument("--exp_name",  default="degrade")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_file = Path("results/logs") / f"{args.exp_name}.log"
    logger = setup_logging(log_file, args.log_level)
    fix_seed(args.seed)

    logger.info("=" * 60)
    logger.info("DEGRADACIÓN LR  (×2 y ×4)")
    logger.info("=" * 60)
    logger.info(f"  input_dir   : {args.input_dir.resolve()}")
    logger.info(f"  output_dir  : {args.output_dir.resolve()}")
    logger.info(f"  degradación : {args.degradation}")
    if args.degradation == "blur_noise":
        logger.info(f"  noise_sigma : {args.noise_sigma}")
        logger.info(f"  blur_kernel : {args.blur_kernel} (x2) / "
                    f"{args.blur_kernel + 2} (x4, escalado automático)")
    logger.info("=" * 60)

    # Comprobar carpeta de entrada
    if not args.input_dir.exists():
        logger.error(f"No existe la carpeta de entrada: {args.input_dir}")
        sys.exit(1)

    image_paths = sorted(
        p for p in args.input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS
    )
    if not image_paths:
        logger.error(f"No hay imágenes en {args.input_dir}")
        sys.exit(1)

    logger.info(f"Imágenes encontradas: {len(image_paths)}\n")

    # Procesar cada factor de escala
    total_ok, total_skip = 0, 0
    for scale in SCALES:
        out_dir = args.output_dir / f"x{scale}"
        logger.info(f"▶ Factor ×{scale}  →  {out_dir}")
        ok, skip = process_scale(
            image_paths=image_paths,
            scale=scale,
            out_dir=out_dir,
            degradation=args.degradation,
            noise_sigma=args.noise_sigma,
            blur_kernel=args.blur_kernel,
            logger=logger,
        )
        logger.info(f"  ✓ {ok} guardadas,  {skip} omitidas\n")
        total_ok += ok
        total_skip += skip

    logger.info("=" * 60)
    logger.info(f"TOTAL  {total_ok} imágenes generadas,  {total_skip} omitidas")
    logger.info(f"Salida: {args.output_dir.resolve()}/")
    logger.info(f"        ├── x2/   ({len(image_paths)} imgs)")
    logger.info(f"        └── x4/   ({len(image_paths)} imgs)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
