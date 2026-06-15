"""
compute_metrics.py — calcula psnr, ssim y lpips entre imágenes sr y sus referencias hr.

guarda por imagen y resumen estadístico en csv.

uso:
    python scripts/compute_metrics.py \
        --sr_dir data/sr/x2/exp0_default \
        --hr_dir data/processed \
        --output_csv results/metrics/sr_exp0_default.csv \
        --metrics psnr ssim lpips \
        --exp_name exp0_default
"""

import argparse
import csv
import logging
import random
import sys
from pathlib import Path

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


def check_cuda() -> "torch.device | None":
    try:
        import torch
        if torch.cuda.is_available():
            logging.info(f"cuda: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        logging.warning("sin cuda")
        return torch.device("cpu")
    except ImportError:
        return None


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def compute_psnr(img_sr: np.ndarray, img_hr: np.ndarray) -> float:
    # mayor psnr = mejor calidad
    from skimage.metrics import peak_signal_noise_ratio
    return float(peak_signal_noise_ratio(img_hr, img_sr, data_range=255))


def compute_ssim(img_sr: np.ndarray, img_hr: np.ndarray) -> float:
    # ssim entre 0 y 1; más cerca de 1 es mejor
    from skimage.metrics import structural_similarity
    if img_sr.ndim == 3:
        return float(structural_similarity(img_hr, img_sr, channel_axis=2, data_range=255))
    else:
        return float(structural_similarity(img_hr, img_sr, data_range=255))


def compute_lpips(img_sr: np.ndarray, img_hr: np.ndarray, lpips_fn, device) -> float:
    # lpips perceptual; menor = más parecido al ojo humano
    import torch
    from torchvision import transforms
    from PIL import Image

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    sr_t = to_tensor(Image.fromarray(img_sr if img_sr.ndim == 3 else np.stack([img_sr]*3, axis=-1))).unsqueeze(0).to(device)
    hr_t = to_tensor(Image.fromarray(img_hr if img_hr.ndim == 3 else np.stack([img_hr]*3, axis=-1))).unsqueeze(0).to(device)
    with torch.no_grad():
        val = lpips_fn(sr_t, hr_t)
    return float(val.item())


def load_image_pair(sr_path: Path, hr_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    # busca el hr con el mismo nombre; redimensiona sr si difieren
    import cv2

    hr_candidates = [
        hr_dir / sr_path.name,
        hr_dir / sr_path.with_suffix(".png").name,
        hr_dir / sr_path.with_suffix(".jpg").name,
    ]
    hr_path = next((c for c in hr_candidates if c.exists()), None)
    if hr_path is None:
        logging.warning(f"  sin hr para: {sr_path.name}")
        return None

    sr = cv2.imread(str(sr_path), cv2.IMREAD_UNCHANGED)
    hr = cv2.imread(str(hr_path), cv2.IMREAD_UNCHANGED)
    if sr is None or hr is None:
        logging.warning(f"  fallo al leer: {sr_path.name}")
        return None

    if sr.shape[:2] != hr.shape[:2]:
        sr = cv2.resize(sr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    if len(sr.shape) == 3 and sr.shape[2] == 3:
        sr = cv2.cvtColor(sr, cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

    return sr, hr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="calcula métricas de calidad sr vs hr (psnr, ssim, lpips)."
    )
    parser.add_argument("--sr_dir", type=Path, required=True)
    parser.add_argument("--hr_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=Path("results/metrics/metrics.csv"))
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "ssim"],
        choices=["psnr", "ssim", "lpips"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--exp_name", default="metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_file = Path("results/logs") / f"{args.exp_name}.log"
    setup_logging(log_file, args.log_level)
    fix_seed(args.seed)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("MÉTRICAS DE IMAGEN")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    device = check_cuda()

    lpips_fn = None
    if "lpips" in args.metrics:
        try:
            import lpips as lpips_lib
            import torch
            device = device or torch.device("cpu")
            lpips_fn = lpips_lib.LPIPS(net="vgg").to(device)
            logger.info("lpips cargado")
        except ImportError:
            logger.warning("lpips no instalado, se omite esa métrica")
            args.metrics = [m for m in args.metrics if m != "lpips"]

    sr_paths = sorted(p for p in args.sr_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not sr_paths:
        logger.error(f"no hay imágenes sr en {args.sr_dir}")
        sys.exit(1)

    logger.info(f"imágenes sr: {len(sr_paths)}")

    fieldnames = ["image"] + args.metrics
    rows = []

    for sr_path in tqdm(sr_paths, desc="calculando métricas", unit="img"):
        pair = load_image_pair(sr_path, args.hr_dir)
        if pair is None:
            continue
        sr_arr, hr_arr = pair

        row = {"image": sr_path.name}
        try:
            if "psnr" in args.metrics:
                row["psnr"] = compute_psnr(sr_arr, hr_arr)
            if "ssim" in args.metrics:
                row["ssim"] = compute_ssim(sr_arr, hr_arr)
            if "lpips" in args.metrics and lpips_fn is not None:
                row["lpips"] = compute_lpips(sr_arr, hr_arr, lpips_fn, device)
        except Exception as e:
            logger.warning(f"  error en {sr_path.name}: {e}")
            continue

        rows.append(row)
        logger.debug(
            f"  {sr_path.name}: " + " | ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "image")
        )

    if not rows:
        logger.error("sin métricas calculadas")
        sys.exit(1)

    # resumen estadístico al final
    logger.info("\n" + "=" * 60)
    logger.info(f"resumen — {args.exp_name}")
    logger.info("=" * 60)
    summary_rows = []
    for metric in args.metrics:
        vals = [r[metric] for r in rows if metric in r]
        mean_val = np.mean(vals)
        std_val  = np.std(vals)
        logger.info(f"  {metric.upper():6s}: {mean_val:.4f} ± {std_val:.4f}")
        summary_rows.append({"image": f"media_{metric}", metric: mean_val})
        summary_rows.append({"image": f"std_{metric}",   metric: std_val})

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        writer.writerows(summary_rows)
    logger.info(f"\nmétricas en: {args.output_csv}")


if __name__ == "__main__":
    main()
