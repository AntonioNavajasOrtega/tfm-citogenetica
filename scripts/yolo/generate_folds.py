"""
generate_folds.py — Genera 5 folds estáticos para K-Fold Cross-Validation con YOLO.

Para cada fold crea la estructura de directorios completa con imágenes y etiquetas
copiadas, más el archivo dataset.yaml listo para ser usado por YOLO.

Estructura generada:
    data/yolo_kfold/
        fold_1/
            train/images/  train/labels/
            val/images/    val/labels/
            dataset.yaml
        fold_2/ ...
        ...

Uso:
    python scripts/yolo/generate_folds.py
"""

import argparse
import json
import logging
import shutil
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
    p = argparse.ArgumentParser(description="Genera folds K-Fold con estructura YOLO completa.")
    p.add_argument("--imgs_dir",       type=Path, default=Path("data/raw/unmarked"),
                   help="Directorio de imágenes de entrada.")
    p.add_argument("--labels_1class_dir", type=Path, default=Path("data/raw/labels_1class"),
                   help="Directorio de etiquetas de 1 clase (para entrenamiento YOLO).")
    p.add_argument("--out_dir",        type=Path, default=Path("data/yolo_kfold"),
                   help="Directorio de salida para los folds.")
    p.add_argument("--folds_json",     type=Path, default=Path("results/folds_info.json"),
                   help="Ruta del JSON de referencia con la asignación de imágenes a folds.")
    p.add_argument("--k_folds",        type=int,  default=5)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--nc",             type=int,  default=1,
                   help="Número de clases YOLO.")
    p.add_argument("--class_names",    nargs="+", default=[""],
                   help="Nombres de las clases YOLO.")
    p.add_argument("--overwrite",      action="store_true",
                   help="Eliminar y recrear out_dir si ya existe.")
    return p.parse_args()


def create_fold_dirs(fold_dir: Path, train_imgs, val_imgs,
                     imgs_dir: Path, labels_dir: Path):
    """Copia imágenes y etiquetas en la estructura train/val esperada por YOLO."""
    for split_name, imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_d = fold_dir / split_name / "images"
        lbl_d = fold_dir / split_name / "labels"
        img_d.mkdir(parents=True, exist_ok=True)
        lbl_d.mkdir(parents=True, exist_ok=True)

        for img_name in imgs:
            src_img = imgs_dir / img_name
            if src_img.exists():
                shutil.copy2(src_img, img_d / img_name)
            else:
                logging.getLogger(__name__).warning(f"Imagen no encontrada: {src_img}")

            lbl_file = Path(img_name).stem + ".txt"
            src_lbl = labels_dir / lbl_file
            if src_lbl.exists():
                shutil.copy2(src_lbl, lbl_d / lbl_file)


def write_yaml(fold_dir: Path, nc: int, class_names: list[str]) -> Path:
    """Escribe el dataset.yaml para el fold."""
    names_str = "[" + ", ".join(f"'{n}'" for n in class_names) + "]"
    yaml_content = (
        f"path: {fold_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: {nc}\n"
        f"names: {names_str}\n"
    )
    yaml_path = fold_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def main():
    args = parse_args()
    logger = setup_logging()

    np.random.seed(args.seed)

    # ── Validaciones de entrada ────────────────────────────────────────────────
    if not args.imgs_dir.exists():
        logger.error(f"El directorio de imágenes no existe: {args.imgs_dir}")
        sys.exit(1)
    if not args.labels_1class_dir.exists():
        logger.error(f"El directorio de etiquetas no existe: {args.labels_1class_dir}")
        sys.exit(1)

    images = sorted([
        p.name for p in args.imgs_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    ])
    if not images:
        logger.error(f"No hay imágenes en {args.imgs_dir}")
        sys.exit(1)
    logger.info(f"Total de imágenes encontradas: {len(images)}")

    # ── Preparar directorio de salida ──────────────────────────────────────────
    if args.out_dir.exists():
        if args.overwrite:
            logger.info(f"Eliminando directorio existente: {args.out_dir}")
            shutil.rmtree(args.out_dir)
        else:
            logger.error(
                f"{args.out_dir} ya existe. Usa --overwrite para sobreescribir."
            )
            sys.exit(1)
    args.out_dir.mkdir(parents=True)

    # ── Generar folds ──────────────────────────────────────────────────────────
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    folds_info = {}

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(images), 1):
        fold_name = f"fold_{fold_idx}"
        fold_dir = args.out_dir / fold_name

        train_imgs = [images[i] for i in train_idx]
        val_imgs   = [images[i] for i in val_idx]

        logger.info(
            f"[{fold_name}] train={len(train_imgs)} imágenes | val={len(val_imgs)} imágenes"
        )

        # Copiar archivos
        create_fold_dirs(fold_dir, train_imgs, val_imgs,
                         args.imgs_dir, args.labels_1class_dir)

        # Escribir YAML
        yaml_path = write_yaml(fold_dir, args.nc, args.class_names)
        logger.info(f"[{fold_name}] dataset.yaml → {yaml_path}")

        folds_info[fold_name] = {"train": train_imgs, "val": val_imgs}

    # ── Guardar JSON de referencia ─────────────────────────────────────────────
    args.folds_json.parent.mkdir(parents=True, exist_ok=True)
    args.folds_json.write_text(json.dumps(folds_info, indent=2), encoding="utf-8")
    logger.info(f"JSON de referencia guardado en: {args.folds_json}")

    logger.info(
        f"✓ {args.k_folds} folds generados en {args.out_dir}"
    )


if __name__ == "__main__":
    main()
