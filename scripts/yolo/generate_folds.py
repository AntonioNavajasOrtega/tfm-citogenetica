"""
generate_folds.py — Genera 5 folds estáticos para las imágenes.

Guarda el resultado en results/folds_info.json para garantizar aislamiento
en todos los pasos posteriores (YOLO, ResShift, CNN).

Uso:
    python scripts/yolo/generate_folds.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np

try:
    from sklearn.model_selection import KFold
except ImportError:
    print("Instala scikit-learn: pip install scikit-learn")
    sys.exit(1)


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--imgs_dir", type=Path, default=Path("data/raw/unmarked"))
    p.add_argument("--out_json", type=Path, default=Path("results/folds_info.json"))
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()
    
    np.random.seed(args.seed)

    images = sorted([p.name for p in args.imgs_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif']])
    if not images:
        logger.error(f"No hay imágenes en {args.imgs_dir}")
        sys.exit(1)

    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    folds_info = {}

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(images), 1):
        train_imgs = [images[i] for i in train_idx]
        val_imgs = [images[i] for i in val_idx]
        folds_info[f"fold_{fold_idx}"] = {"train": train_imgs, "val": val_imgs}

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(folds_info, indent=2))
    
    logger.info(f"Se generaron {args.k_folds} folds con {len(images)} imágenes en total.")
    logger.info(f"Archivo guardado en: {args.out_json}")


if __name__ == "__main__":
    main()
