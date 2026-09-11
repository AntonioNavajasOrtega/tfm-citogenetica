"""
split_dataset.py — divide las 50 imágenes procesadas en train/val con split fijo.

genera data/train/ y data/val/ con imágenes y etiquetas.
guarda el split en results/splits.json para reproducibilidad.
genera el archivo dataset.yaml requerido por YOLO.

uso:
    python scripts/yolo/split_dataset.py --seed 42 --val_size 5
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="divide processed en train/val")
    p.add_argument("--processed_dir", type=Path, default=Path("data/raw/unmarked"))
    p.add_argument("--labels_dir",    type=Path, default=Path("data/raw/labels_1class"))
    p.add_argument("--out_dir",       type=Path, default=Path("data"))
    p.add_argument("--val_size",      type=int,  default=5)
    p.add_argument("--seed",          type=int,  default=42)
    p.add_argument("--nc",            type=int,  default=1, help="Número de clases")
    p.add_argument("--class_names",   type=str,  default='["chromosome"]', help="Nombres de clases en formato lista python string")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # recoger todas las imágenes disponibles
    images = sorted(
        p for p in args.processed_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )
    if len(images) < args.val_size + 1:
        print(f"solo hay {len(images)} imágenes, no se puede hacer split")
        sys.exit(1)

    # split aleatorio con semilla fija
    rng = random.Random(args.seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    val_imgs   = shuffled[:args.val_size]
    train_imgs = shuffled[args.val_size:]

    print(f"train: {len(train_imgs)} imágenes | val: {len(val_imgs)} imágenes")

    # crear carpetas destino
    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_dir = args.out_dir / split_name / "images"
        lbl_dir = args.out_dir / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in split_imgs:
            # copiar imagen
            shutil.copy2(img_path, img_dir / img_path.name)

            # copiar etiqueta si existe
            lbl_src = args.labels_dir / img_path.with_suffix(".txt").name
            if lbl_src.exists():
                shutil.copy2(lbl_src, lbl_dir / lbl_src.name)
            else:
                print(f"  aviso: sin etiqueta para {img_path.name}")

    # guardar info del split para poder reproducirlo
    split_info = {
        "seed": args.seed,
        "train": [p.name for p in train_imgs],
        "val":   [p.name for p in val_imgs],
    }
    splits_path = Path("results/splits.json")
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(split_info, indent=2, ensure_ascii=False))
    print(f"split guardado en {splits_path}")
    
    # generar dataset.yaml
    yaml_content = (
        f"path: {args.out_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: {args.nc}\n"
        f"names: {args.class_names}\n"
    )
    yaml_path = args.out_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"yaml generado en {yaml_path}")

    print("listo")


if __name__ == "__main__":
    main()
