"""
evaluate_yolo.py — evalúa un checkpoint yolo sobre cualquier directorio de imágenes.

sirve para comparar mAP en tres condiciones:
  - modo 'original': imágenes test sin SR (baseline "no operation")
  - modo 'sr':       imágenes test con SR aplicada
  - modo 'custom':   cualquier directorio que se pase

uso:
    # baseline (no operation)
    python scripts/evaluate_yolo.py \
        --img_dir data/test/images \
        --lbl_dir data/test/labels \
        --checkpoint models/yolo_baseline/best.pt \
        --exp_name no_operation

    # sobre imágenes sr
    python scripts/evaluate_yolo.py \
        --img_dir data/sr/sr_ddim_50 \
        --lbl_dir data/test/labels \
        --checkpoint models/yolo_baseline/best.pt \
        --exp_name sr_ddim_50
"""

import argparse
import csv
import logging
import shutil
import sys
import tempfile
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("instala ultralytics: pip install ultralytics")
    sys.exit(1)


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="evalúa yolo sobre un directorio de imágenes")
    p.add_argument("--img_dir",     type=Path, required=True,
                   help="directorio con imágenes a evaluar")
    p.add_argument("--lbl_dir",     type=Path, required=True,
                   help="directorio con etiquetas yolo (.txt)")
    p.add_argument("--checkpoint",  type=Path, default=Path("models/yolo_baseline/best.pt"),
                   help="checkpoint yolo a usar")
    p.add_argument("--img_size",    type=int, default=512)
    p.add_argument("--device",      default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--exp_name",    default=None,
                   help="nombre del experimento para el csv de salida")
    p.add_argument("--output_dir",  type=Path, default=Path("results/metrics"))
    p.add_argument("--log_level",   default="INFO")
    return p.parse_args()


def evaluate(
    img_dir: Path,
    lbl_dir: Path,
    checkpoint: Path,
    img_size: int,
    device: str,
    exp_name: str,
) -> dict:
    logger = logging.getLogger(__name__)

    if not checkpoint.exists():
        logger.error(f"checkpoint no encontrado: {checkpoint}")
        sys.exit(1)

    img_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not img_paths:
        logger.error(f"no hay imágenes en {img_dir}")
        sys.exit(1)

    logger.info(f"imágenes: {len(img_paths)} en {img_dir}")

    with tempfile.TemporaryDirectory(prefix="yolo_eval_") as tmp:
        tmp_dir = Path(tmp)
        img_d = tmp_dir / "images"
        lbl_d = tmp_dir / "labels"
        img_d.mkdir()
        lbl_d.mkdir()

        for img_path in img_paths:
            shutil.copy2(img_path, img_d / img_path.name)
            # las etiquetas siempre vienen del test original (mismas para sr y original)
            lbl_src = lbl_dir / img_path.stem
            # buscar el .txt con el mismo stem (puede ser stem diferente si sr renombra)
            lbl_candidate = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_candidate.exists():
                # buscar por nombre base por si el sr añade sufijos
                candidates = list(lbl_dir.glob(f"{img_path.stem}*.txt"))
                lbl_candidate = candidates[0] if candidates else None
            if lbl_candidate and lbl_candidate.exists():
                shutil.copy2(lbl_candidate, lbl_d / f"{img_path.stem}.txt")
            else:
                logger.warning(f"  sin etiqueta para {img_path.stem}")

        yaml_path = tmp_dir / "eval.yaml"
        yaml_path.write_text(
            f"path: {tmp_dir}\n"
            f"val: images\n"
            f"nc: 2\n"
            f"names: ['chromosome', 'dicentric']\n"
        )

        model = YOLO(str(checkpoint))
        metrics = model.val(
            data=str(yaml_path),
            imgsz=img_size,
            device=device,
            verbose=True,
        )

    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    return {
        "exp":       exp_name,
        "img_dir":   str(img_dir),
        "n_imgs":    len(img_paths),
        "map50":     float(metrics.box.map50),
        "map50_95":  float(metrics.box.map),
        "precision": mp,
        "recall":    mr,
        "f1":        2 * mp * mr / (mp + mr + 1e-8),
    }


def main() -> None:
    args = parse_args()

    # nombre por defecto es el nombre del directorio de imágenes
    exp_name = args.exp_name or args.img_dir.name

    log_file = Path("results/logs") / f"eval_{exp_name}.log"
    logger = setup_logging(log_file, args.log_level)

    logger.info("=" * 60)
    logger.info("EVALUACIÓN YOLO")
    logger.info("=" * 60)
    logger.info(f"  exp_name:   {exp_name}")
    logger.info(f"  img_dir:    {args.img_dir}")
    logger.info(f"  lbl_dir:    {args.lbl_dir}")
    logger.info(f"  checkpoint: {args.checkpoint}")
    logger.info("=" * 60)

    result = evaluate(
        img_dir=args.img_dir,
        lbl_dir=args.lbl_dir,
        checkpoint=args.checkpoint,
        img_size=args.img_size,
        device=args.device,
        exp_name=exp_name,
    )

    # mostrar resultado
    logger.info("=" * 60)
    logger.info(f"map50:     {result['map50']:.4f}")
    logger.info(f"map50-95:  {result['map50_95']:.4f}")
    logger.info(f"precision: {result['precision']:.4f}")
    logger.info(f"recall:    {result['recall']:.4f}")
    logger.info(f"f1:        {result['f1']:.4f}")
    logger.info("=" * 60)

    # guardar csv acumulativo (append si ya existe)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "yolo_comparison.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        fieldnames = ["exp", "img_dir", "n_imgs", "map50", "map50_95", "precision", "recall", "f1"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(result)
    logger.info(f"resultado añadido a: {csv_path}")


if __name__ == "__main__":
    main()
