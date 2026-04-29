"""
split_dataset.py — divide las 50 imágenes procesadas en train/test con split fijo.

genera data/train/ y data/test/ con imágenes y etiquetas.
guarda el split en results/splits.json para reproducibilidad.

uso:
    python scripts/split_dataset.py --seed 42 --test_size 10
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="divide processed en train/test")
    p.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--labels_dir",    type=Path, default=Path("data/raw/labels"))
    p.add_argument("--out_dir",       type=Path, default=Path("data"))
    p.add_argument("--test_size",     type=int,  default=10)
    p.add_argument("--seed",          type=int,  default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # recoger todas las imágenes disponibles
    images = sorted(
        p for p in args.processed_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )
    if len(images) < args.test_size + 1:
        print(f"solo hay {len(images)} imágenes, no se puede hacer split")
        sys.exit(1)

    # split aleatorio con semilla fija
    rng = random.Random(args.seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    test_imgs  = shuffled[:args.test_size]
    train_imgs = shuffled[args.test_size:]

    print(f"train: {len(train_imgs)} imágenes | test: {len(test_imgs)} imágenes")

    # crear carpetas destino
    for split_name, split_imgs in [("train", train_imgs), ("test", test_imgs)]:
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
        "test":  [p.name for p in test_imgs],
    }
    splits_path = Path("results/splits.json")
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(split_info, indent=2, ensure_ascii=False))
    print(f"split guardado en {splits_path}")

    print("listo")


if __name__ == "__main__":
    main()
