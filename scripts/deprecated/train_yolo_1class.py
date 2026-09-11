"""
train_yolo_1class.py — entrena yolov11n para detectar cromosomas (1 clase) en un split fijo.

Genera un dataset.yaml dinámicamente y entrena sin K-Fold.

uso:
    python scripts/yolo/train_yolo_1class.py --epochs 150 --exp_name yolo_1class
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("instala ultralytics: pip install ultralytics")
    sys.exit(1)


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


def create_yaml(data_dir: Path, nc: int, class_names: list[str]) -> Path:
    yaml_content = (
        f"path: {data_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: test/images\n" # Asumiendo que test se usa como val en este split fijo
        f"nc: {nc}\n"
        f"names: {class_names}\n"
    )
    yaml_path = data_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="entrenamiento yolov11 1 clase")
    p.add_argument("--data_dir", type=Path, default=Path("data/yolo_1class"))
    p.add_argument("--output_dir", type=Path, default=Path("models/yolo_1class"))
    p.add_argument("--model_size", default="m", choices=["n", "s", "m"])
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--img_size", type=int, default=1280)
    p.add_argument("--batch_size", type=int, default=-1)
    p.add_argument("--freeze", type=int, default=10)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_level", default="INFO")
    p.add_argument("--exp_name", default="yolo_1class")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_file = Path("results/logs") / f"{args.exp_name}.log"
    logger = setup_logging(log_file, args.log_level)

    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO YOLO 1 CLASE")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    if not args.data_dir.exists():
        logger.error(f"Error: la carpeta {args.data_dir} no existe. Ejecuta split_dataset.py primero.")
        sys.exit(1)

    yaml_path = create_yaml(args.data_dir, nc=1, class_names=["chromosome"])

    project_dir = (args.output_dir / args.exp_name).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        if torch.cuda.is_available() and args.device == "cuda":
            device = "0"
        else:
            device = "cpu"
    except ImportError:
        device = "cpu"

    model = YOLO(f"yolo11{args.model_size}.pt")
    
    logger.info(f"Iniciando entrenamiento en {device}...")
    model.train(
        data=str(yaml_path.resolve()),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=device,
        project=str(project_dir),
        name="train",
        exist_ok=True,
        seed=args.seed,
        patience=args.patience,
        lr0=0.005,
        lrf=0.01,
        weight_decay=0.001,
        warmup_epochs=5,
        close_mosaic=20,
        freeze=args.freeze,
        cache=False,
        workers=args.workers,
        plots=True,
    )
    
    logger.info("Entrenamiento finalizado. El mejor modelo se encuentra en:")
    logger.info(str(project_dir / "train/weights/best.pt"))


if __name__ == "__main__":
    main()
