"""
train_yolo.py — entrena yolov11 con validación cruzada k-fold o leave-one-out.

al terminar genera un resumen por consola y un csv con métricas por fold.

uso:
    python scripts/train_yolo.py \
        --data_dir data/processed \
        --annot_dir data/raw/labels \
        --output_dir models/yolo_baseline \
        --model_size n \
        --cv_mode kfold \
        --k_folds 5 \
        --epochs 50 \
        --exp_name baseline_kfold5
"""

import argparse
import csv
import logging
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("instala ultralytics: pip install ultralytics")
    sys.exit(1)


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
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def check_cuda() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logging.info(f"cuda: {name}")
            return "cuda"
        else:
            logging.warning("sin cuda, muy lento en cpu")
            return "cpu"
    except ImportError:
        logging.warning("pytorch no disponible")
        return "cpu"


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def get_image_paths(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)


def build_kfold_splits(paths: list[Path], k: int, seed: int) -> list[tuple[list, list]]:
    # reparto aleatorio en k particiones
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    folds = []
    n = len(shuffled)
    fold_size = n // k
    for i in range(k):
        val = shuffled[i * fold_size: (i + 1) * fold_size] if i < k - 1 else shuffled[i * fold_size:]
        train = [p for p in shuffled if p not in val]
        folds.append((train, val))
    return folds


def build_loo_splits(paths: list[Path], loo_n: int, seed: int) -> list[tuple[list, list]]:
    # leave-one-out sobre un subconjunto de tamaño loo_n
    rng = random.Random(seed)
    subset = paths[:]
    rng.shuffle(subset)
    subset = subset[:loo_n]
    folds = []
    for i, val_img in enumerate(subset):
        train = [p for p in subset if p != val_img]
        folds.append((train, [val_img]))
    return folds


def train_fold(
    fold_idx: int,
    train_paths: list[Path],
    val_paths: list[Path],
    annot_dir: Path,
    output_dir: Path,
    model_size: str,
    epochs: int,
    img_size: int,
    batch_size: int,
    device: str,
    seed: int,
    exp_name: str,
) -> dict:
    logger = logging.getLogger(__name__)
    logger.info(f"\n--- fold {fold_idx} | train={len(train_paths)} | val={len(val_paths)} ---")

    with tempfile.TemporaryDirectory(prefix=f"yolo_fold{fold_idx}_") as tmp:
        tmp_dir = Path(tmp)

        # copia imágenes y etiquetas al directorio temporal del fold
        for split_name, split_paths in [("train", train_paths), ("val", val_paths)]:
            img_dir = tmp_dir / split_name / "images"
            lbl_dir = tmp_dir / split_name / "labels"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            for img_path in split_paths:
                shutil.copy2(img_path, img_dir / img_path.name)
                lbl_src = annot_dir / img_path.with_suffix(".txt").name
                if lbl_src.exists():
                    shutil.copy2(lbl_src, lbl_dir / lbl_src.name)
                else:
                    logger.warning(f"  sin anotación: {img_path.name}")

        yaml_path = tmp_dir / "dataset.yaml"
        yaml_path.write_text(
            f"path: {tmp_dir}\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"nc: 1\n"
            f"names: ['chromosome']\n"
        )

        fold_out = output_dir / f"{exp_name}_fold{fold_idx}"
        fold_out.mkdir(parents=True, exist_ok=True)

        model = YOLO(f"yolo11{model_size}.pt")
        model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            project=str(fold_out),
            name="train",
            exist_ok=True,
            seed=seed,
            verbose=False,
        )

        metrics = model.val(
            data=str(yaml_path),
            imgsz=img_size,
            device=device,
            verbose=False,
        )

    result = {
        "fold": fold_idx,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "f1": 2 * float(metrics.box.mp) * float(metrics.box.mr)
              / (float(metrics.box.mp) + float(metrics.box.mr) + 1e-8),
    }
    logger.info(
        f"  map50={result['map50']:.4f} | map50-95={result['map50_95']:.4f} "
        f"| p={result['precision']:.4f} | r={result['recall']:.4f} | f1={result['f1']:.4f}"
    )
    return result


def print_summary(results: list[dict], exp_name: str, csv_path: Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 70)
    logger.info(f"resumen — {exp_name}")
    logger.info("=" * 70)
    header = f"{'fold':>6} {'map50':>8} {'map50-95':>10} {'precision':>10} {'recall':>8} {'f1':>8}"
    logger.info(header)
    logger.info("-" * 70)

    keys = ["map50", "map50_95", "precision", "recall", "f1"]
    for r in results:
        logger.info(
            f"{r['fold']:>6} {r['map50']:>8.4f} {r['map50_95']:>10.4f} "
            f"{r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f}"
        )

    logger.info("-" * 70)
    means = {k: np.mean([r[k] for r in results]) for k in keys}
    stds  = {k: np.std([r[k] for r in results]) for k in keys}
    logger.info(
        f"{'media':>6} {means['map50']:>8.4f} {means['map50_95']:>10.4f} "
        f"{means['precision']:>10.4f} {means['recall']:>8.4f} {means['f1']:>8.4f}"
    )
    logger.info(
        f"{'std':>6} {stds['map50']:>8.4f} {stds['map50_95']:>10.4f} "
        f"{stds['precision']:>10.4f} {stds['recall']:>8.4f} {stds['f1']:>8.4f}"
    )
    logger.info("=" * 70)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold"] + keys)
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({"fold": "media", **means})
        writer.writerow({"fold": "std", **stds})
    logger.info(f"métricas en: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="entrenamiento yolov11 con validación cruzada."
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--annot_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("models/yolo_baseline"))
    parser.add_argument("--model_size", default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cv_mode", default="kfold", choices=["kfold", "loo"])
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--loo_n", type=int, default=20)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--exp_name", default="baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_file = Path("results/logs") / f"{args.exp_name}.log"
    setup_logging(log_file, args.log_level)
    fix_seed(args.seed)
    device = check_cuda() if args.device == "cuda" else "cpu"

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO YOLO BASELINE")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = get_image_paths(args.data_dir)
    if not image_paths:
        logger.error(f"no hay imágenes en {args.data_dir}")
        sys.exit(1)
    logger.info(f"imágenes: {len(image_paths)}")

    if args.cv_mode == "kfold":
        splits = build_kfold_splits(image_paths, args.k_folds, args.seed)
        logger.info(f"modo: k-fold (k={args.k_folds})")
    else:
        splits = build_loo_splits(image_paths, args.loo_n, args.seed)
        logger.info(f"modo: loo (n={args.loo_n})")

    results = []
    for fold_idx, (train_paths, val_paths) in enumerate(
        tqdm(splits, desc="folds", unit="fold"), start=1
    ):
        result = train_fold(
            fold_idx=fold_idx,
            train_paths=train_paths,
            val_paths=val_paths,
            annot_dir=args.annot_dir,
            output_dir=args.output_dir,
            model_size=args.model_size,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            exp_name=args.exp_name,
        )
        results.append(result)

    csv_path = Path("results/metrics") / f"{args.exp_name}.csv"
    print_summary(results, args.exp_name, csv_path)


if __name__ == "__main__":
    main()
