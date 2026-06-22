"""
train_yolo_kfold.py — Entrena YOLO (1 clase) en los 5 folds estáticos.

Requiere haber ejecutado generate_folds.py primero.

Uso:
    python scripts/yolo/train_yolo_kfold.py --epochs 100
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Instala ultralytics: pip install ultralytics")
    sys.exit(1)


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger(__name__)


def create_fold_dataset(fold_idx, train_imgs, val_imgs, imgs_dir, labels_1class_dir, out_base_dir):
    fold_dir = out_base_dir / f"fold_{fold_idx}"
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
        
    for split_name, imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_d = fold_dir / split_name / "images"
        lbl_d = fold_dir / split_name / "labels"
        img_d.mkdir(parents=True)
        lbl_d.mkdir(parents=True)
        for img in imgs:
            shutil.copy2(imgs_dir / img, img_d / img)
            lbl_file = img.replace(Path(img).suffix, ".txt")
            lbl_src = labels_1class_dir / lbl_file
            if lbl_src.exists():
                shutil.copy2(lbl_src, lbl_d / lbl_file)

    yaml_content = f"""path: {fold_dir.resolve()}
train: train/images
val: val/images
nc: 1
names: ['chromosome']
"""
    yaml_path = fold_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folds_json", type=Path, default=Path("results/folds_info.json"))
    p.add_argument("--imgs_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--labels_1class_dir", type=Path, default=Path("data/raw/labels_1class"))
    p.add_argument("--yolo_data_dir", type=Path, default=Path("data/yolo_kfold"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="0", help="cuda device (ej. 0) o 'cpu'")
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    if not args.folds_json.exists():
        logger.error(f"Falta {args.folds_json}. Ejecuta generate_folds.py primero.")
        sys.exit(1)

    folds_info = json.loads(args.folds_json.read_text())

    for fold_name, f_data in folds_info.items():
        fold_idx = fold_name.replace("fold_", "")
        logger.info("=" * 60)
        logger.info(f"ENTRENANDO YOLO - FOLD {fold_idx}")
        logger.info("=" * 60)
        
        # 1. Crear dataset YAML
        yaml_path = create_fold_dataset(
            fold_idx, f_data["train"], f_data["val"], 
            args.imgs_dir, args.labels_1class_dir, args.yolo_data_dir
        )
        
        # 2. Entrenar YOLO
        model = YOLO("yolo11n.pt")
        project_dir = Path("models/yolo_kfold") / fold_name
        
        model.train(
            data=str(yaml_path.resolve()),
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch_size,
            device=args.device,
            project=str(project_dir),
            name="train",
            exist_ok=True,
            seed=args.seed,
            verbose=False,
            plots=True
        )

    logger.info("Entrenamiento K-Fold YOLO completado.")

if __name__ == "__main__":
    main()
