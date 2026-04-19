"""
06_evaluate_yolo.py — Evaluación YOLO sobre imágenes SR.

Compara el rendimiento del detector YOLO (entrenado sobre originales)
cuando se aplica sobre imágenes con super-resolución.

Nota: las anotaciones YOLO en coordenadas normalizadas [0,1] son válidas
aunque SR amplíe la resolución, ya que no cambian las proporciones.

Uso:
    python scripts/06_evaluate_yolo.py \
        --sr_dir data/sr/x2/exp0_default \
        --annot_dir data/annotations \
        --yolo_checkpoint models/yolo_baseline/baseline_kfold5_fold1/train/weights/best.pt \
        --output_dir results/metrics \
        --exp_name yolo_on_sr_exp0
"""

import argparse
import csv
import logging
import random
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics no instalado.")
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


def check_cuda() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            logging.info(f"CUDA disponible: {torch.cuda.get_device_name(0)}")
            return "cuda"
        logging.warning("CUDA NO disponible.")
        return "cpu"
    except ImportError:
        return "cpu"


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación de YOLO sobre imágenes con super-resolución."
    )
    parser.add_argument("--sr_dir", type=Path, required=True)
    parser.add_argument("--annot_dir", type=Path, required=True)
    parser.add_argument("--yolo_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--conf_threshold", type=float, default=0.25)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--exp_name", default="yolo_on_sr")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_file = Path("results/logs") / f"{args.exp_name}.log"
    setup_logging(log_file, args.log_level)
    fix_seed(args.seed)
    device = check_cuda() if args.device == "cuda" else "cpu"

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("SCRIPT 06 — EVALUACIÓN YOLO SOBRE SR")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    # Verificar checkpoint
    if not args.yolo_checkpoint.exists():
        logger.error(f"Checkpoint YOLO no encontrado: {args.yolo_checkpoint}")
        sys.exit(1)

    # Recopilar imágenes SR
    sr_paths = sorted(p for p in args.sr_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not sr_paths:
        logger.error(f"No se encontraron imágenes SR en {args.sr_dir}.")
        sys.exit(1)
    logger.info(f"Imágenes SR encontradas: {len(sr_paths)}")

    # Construir dataset temporal para evaluación con Ultralytics
    with tempfile.TemporaryDirectory(prefix="yolo_eval_") as tmp:
        tmp_dir = Path(tmp)
        img_dir = tmp_dir / "images"
        lbl_dir = tmp_dir / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        for sr_path in sr_paths:
            shutil.copy2(sr_path, img_dir / sr_path.name)
            annot = args.annot_dir / sr_path.with_suffix(".txt").name
            if annot.exists():
                shutil.copy2(annot, lbl_dir / annot.name)
            else:
                logger.warning(f"  Sin anotación para: {sr_path.name}")
                # Crear archivo vacío (imagen sin objetos)
                (lbl_dir / sr_path.with_suffix(".txt").name).touch()

        yaml_path = tmp_dir / "eval.yaml"
        yaml_path.write_text(
            f"path: {tmp_dir}\n"
            f"val: images\n"
            f"nc: 1\n"
            f"names: ['chromosome']\n"
        )

        model = YOLO(str(args.yolo_checkpoint))
        metrics = model.val(
            data=str(yaml_path),
            imgsz=args.img_size,
            conf=args.conf_threshold,
            iou=args.iou_threshold,
            device=device,
            verbose=True,
        )

    # Extraer métricas
    map50     = float(metrics.box.map50)
    map50_95  = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall    = float(metrics.box.mr)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    logger.info("\n" + "=" * 50)
    logger.info(f"RESULTADOS — {args.exp_name}")
    logger.info(f"  mAP50:     {map50:.4f}")
    logger.info(f"  mAP50-95:  {map50_95:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1:        {f1:.4f}")
    logger.info("=" * 50)

    # Guardar CSV
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.exp_name}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["exp_name", "map50", "map50_95", "precision", "recall", "f1"]
        )
        writer.writeheader()
        writer.writerow({
            "exp_name": args.exp_name,
            "map50": map50,
            "map50_95": map50_95,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    logger.info(f"Métricas guardadas en: {csv_path}")


if __name__ == "__main__":
    main()
