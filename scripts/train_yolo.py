"""
train_yolo.py — entrena yolov11n con validación cruzada 5-fold sobre train set.

evalúa el mejor fold sobre el test set fijo (no visto durante entrenamiento).
genera el baseline "no operation" del paper de referencia.

uso:
    python scripts/train_yolo.py --epochs 150 --exp_name baseline_kfold5
"""

import argparse
import csv
import logging
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

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
            logging.info(f"cuda: {torch.cuda.get_device_name(0)}")
            return "0"  # ultralytics espera el índice como string
        logging.warning("sin cuda")
        return "cpu"
    except ImportError:
        return "cpu"


def get_image_paths(img_dir: Path) -> list[Path]:
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)


def build_kfold_splits(paths: list[Path], k: int, seed: int) -> list[tuple[list, list]]:
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    folds = []
    for i in range(k):
        start = i * (n // k)
        end = (i + 1) * (n // k) if i < k - 1 else n
        val   = shuffled[start:end]
        train = [p for p in shuffled if p not in val]
        folds.append((train, val))
    return folds


def prepare_fold_dir(
    fold_idx: int,
    train_imgs: list[Path],
    val_imgs: list[Path],
    labels_dir: Path,
    nc: int,
    class_names: list[str],
) -> Path:
    """
    crea la estructura de carpetas para un fold.
    usa data/folds/foldN/ para que sea persistente y sin path muy largo.
    """
    fold_dir = Path("data/folds") / f"fold{fold_idx}"
    if fold_dir.exists():
        shutil.rmtree(fold_dir)

    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_d = fold_dir / split_name / "images"
        lbl_d = fold_dir / split_name / "labels"
        img_d.mkdir(parents=True)
        lbl_d.mkdir(parents=True)
        for img_path in split_imgs:
            shutil.copy2(img_path, img_d / img_path.name)
            lbl = labels_dir / img_path.with_suffix(".txt").name
            if lbl.exists():
                shutil.copy2(lbl, lbl_d / lbl.name)

    # yaml con path absoluto para evitar que ultralytics añada prefijos raros
    yaml_content = (
        f"path: {fold_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: {nc}\n"
        f"names: {class_names}\n"
    )
    (fold_dir / "dataset.yaml").write_text(yaml_content)
    return fold_dir


def train_fold(
    fold_idx: int,
    fold_dir: Path,
    output_dir: Path,
    model_size: str,
    epochs: int,
    patience: int,
    img_size: int,
    batch_size: int,
    device: str,
    seed: int,
    exp_name: str,
    freeze: int,
    workers: int = 0,
) -> dict:
    logger = logging.getLogger(__name__)
    n_train = len(list((fold_dir / "train/images").iterdir()))
    n_val   = len(list((fold_dir / "val/images").iterdir()))
    logger.info(f"\n--- fold {fold_idx} | train={n_train} val={n_val} ---")

    # paths absolutos para evitar el prefijo runs/detect/ de ultralytics
    project_dir = (output_dir / f"{exp_name}_fold{fold_idx}").resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(f"yolo11{model_size}.pt")
    model.train(
        data=str((fold_dir / "dataset.yaml").resolve()),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=str(project_dir),
        name="train",
        exist_ok=True,
        seed=seed,
        verbose=False,
        # hiperparámetros para dataset pequeño
        patience=patience,            # early stopping tras N épocas sin mejora
        lr0=0.005,              # lr inicial menor que el default (0.01) — más estable
        lrf=0.01,               # lr final = lr0 * lrf
        weight_decay=0.001,     # más regularización
        warmup_epochs=5,        # más warmup
        close_mosaic=20,        # desactivar mosaic en los últimos 20 epochs
        freeze=freeze,          # congelar primeras N capas del backbone
        cache=False,            # no cachear (pocas imgs, no compensa memoria)
        workers=workers,        # 0 en windows para evitar error de shared memory
        plots=True,
    )

    # validar sobre val set del fold
    metrics = model.val(
        data=str((fold_dir / "dataset.yaml").resolve()),
        imgsz=img_size,
        device=device,
        verbose=False,
    )

    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    weights_path = project_dir / "train" / "weights" / "best.pt"
    result = {
        "fold":      fold_idx,
        "map50":     float(metrics.box.map50),
        "map50_95":  float(metrics.box.map),
        "precision": mp,
        "recall":    mr,
        "f1":        2 * mp * mr / (mp + mr + 1e-8),
        "weights":   str(weights_path),
    }
    logger.info(
        f"  map50={result['map50']:.4f} | prec={result['precision']:.4f} "
        f"| rec={result['recall']:.4f} | f1={result['f1']:.4f}"
    )
    return result


def evaluate_on_test(
    best_weights: str,
    test_img_dir: Path,
    test_lbl_dir: Path,
    img_size: int,
    device: str,
    exp_name: str,
) -> dict:
    """
    evaluación final sobre el test set fijo.
    este número es el "no operation" comparable con el paper de referencia.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n=== test set ({test_img_dir}) ===")

    # crear dir temporal persistente para el test yaml
    test_data_dir = Path("data/folds/test")
    if test_data_dir.exists():
        shutil.rmtree(test_data_dir)

    img_d = test_data_dir / "images"
    lbl_d = test_data_dir / "labels"
    img_d.mkdir(parents=True)
    lbl_d.mkdir(parents=True)

    for img_path in get_image_paths(test_img_dir):
        shutil.copy2(img_path, img_d / img_path.name)
        lbl = test_lbl_dir / img_path.with_suffix(".txt").name
        if lbl.exists():
            shutil.copy2(lbl, lbl_d / lbl.name)

    yaml_path = test_data_dir / "test.yaml"
    yaml_path.write_text(
        f"path: {test_data_dir.resolve()}\n"
        f"val: images\n"
        f"nc: 2\n"
        f"names: ['chromosome', 'dicentric']\n"
    )

    model = YOLO(best_weights)
    metrics = model.val(
        data=str(yaml_path.resolve()),
        imgsz=img_size,
        device=device,
        verbose=True,
    )

    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    result = {
        "exp":       exp_name,
        "split":     "test",
        "map50":     float(metrics.box.map50),
        "map50_95":  float(metrics.box.map),
        "precision": mp,
        "recall":    mr,
        "f1":        2 * mp * mr / (mp + mr + 1e-8),
    }
    logger.info(f"TEST — map50={result['map50']:.4f} | f1={result['f1']:.4f}")
    return result


def print_cv_summary(results: list[dict], exp_name: str, csv_path: Path) -> None:
    logger = logging.getLogger(__name__)
    keys = ["map50", "map50_95", "precision", "recall", "f1"]
    logger.info("\n" + "=" * 65)
    logger.info(f"resumen 5-fold cv — {exp_name}")
    logger.info("=" * 65)
    logger.info(f"{'fold':>5} {'map50':>8} {'map50-95':>10} {'prec':>8} {'rec':>8} {'f1':>8}")
    logger.info("-" * 65)
    for r in results:
        logger.info(f"{r['fold']:>5} {r['map50']:>8.4f} {r['map50_95']:>10.4f} "
                    f"{r['precision']:>8.4f} {r['recall']:>8.4f} {r['f1']:>8.4f}")
    means = {k: float(np.mean([r[k] for r in results])) for k in keys}
    stds  = {k: float(np.std( [r[k] for r in results])) for k in keys}
    logger.info("-" * 65)
    logger.info(f"{'media':>5} {means['map50']:>8.4f} {means['map50_95']:>10.4f} "
                f"{means['precision']:>8.4f} {means['recall']:>8.4f} {means['f1']:>8.4f}")
    logger.info(f"{'std':>5}  {stds['map50']:>8.4f} {stds['map50_95']:>10.4f} "
                f"{stds['precision']:>8.4f} {stds['recall']:>8.4f} {stds['f1']:>8.4f}")
    logger.info("=" * 65)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold"] + keys)
        w.writeheader()
        w.writerows([{k: r[k] for k in ["fold"] + keys} for r in results])
        w.writerow({"fold": "media", **means})
        w.writerow({"fold": "std",   **stds})
    logger.info(f"métricas cv en: {csv_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="entrenamiento yolov11 con 5-fold cv")
    p.add_argument("--train_img_dir", type=Path, default=Path("data/train/images"))
    p.add_argument("--train_lbl_dir", type=Path, default=Path("data/train/labels"))
    p.add_argument("--test_img_dir",  type=Path, default=Path("data/test/images"))
    p.add_argument("--test_lbl_dir",  type=Path, default=Path("data/test/labels"))
    p.add_argument("--output_dir",    type=Path, default=Path("models/yolo_baseline"))
    p.add_argument("--model_size",    default="n", choices=["n", "s", "m"],
                   help="nano recomendado para datasets pequeños")
    p.add_argument("--epochs",        type=int, default=150)
    p.add_argument("--patience",      type=int, default=40)
    p.add_argument("--img_size",      type=int, default=512)
    p.add_argument("--batch_size",    type=int, default=4)
    p.add_argument("--k_folds",       type=int, default=5)
    p.add_argument("--freeze",        type=int, default=10,
                   help="capas del backbone a congelar (0=ninguna, 10=todo el backbone)")
    p.add_argument("--workers",       type=int, default=0,
                   help="workers del dataloader (0=sin multiprocessing, fix para windows)")
    p.add_argument("--device",        default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--log_level",     default="INFO")
    p.add_argument("--exp_name",      default="baseline_kfold5")
    p.add_argument("--nc",            type=int, default=2)
    p.add_argument("--class_names",   nargs="+", default=["chromosome", "dicentric"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_file = Path("results/logs") / f"{args.exp_name}.log"
    logger = setup_logging(log_file, args.log_level)
    fix_seed(args.seed)
    device = check_cuda() if args.device == "cuda" else "cpu"

    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO YOLO — BASELINE")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    train_imgs = get_image_paths(args.train_img_dir)
    if not train_imgs:
        logger.error(f"no hay imágenes en {args.train_img_dir}")
        sys.exit(1)
    logger.info(f"imágenes train: {len(train_imgs)}")

    splits = build_kfold_splits(train_imgs, args.k_folds, args.seed)
    cv_results = []

    for fold_idx, (train_paths, val_paths) in enumerate(
        tqdm(splits, desc="5-fold cv", unit="fold"), start=1
    ):
        fold_dir = prepare_fold_dir(
            fold_idx, train_paths, val_paths, args.train_lbl_dir,
            args.nc, args.class_names,
        )
        r = train_fold(
            fold_idx=fold_idx,
            fold_dir=fold_dir,
            output_dir=args.output_dir,
            model_size=args.model_size,
            epochs=args.epochs,
            patience=args.patience,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            exp_name=args.exp_name,
            freeze=args.freeze,
            workers=args.workers,
        )
        cv_results.append(r)

    csv_cv = Path("results/metrics") / f"{args.exp_name}_cv.csv"
    print_cv_summary(cv_results, args.exp_name, csv_cv)

    # elegir el mejor fold y evaluar sobre test
    best = max(cv_results, key=lambda r: r["map50"])
    logger.info(f"\nmejor fold: {best['fold']} (map50={best['map50']:.4f})")

    if args.test_img_dir.exists():
        test_result = evaluate_on_test(
            best_weights=best["weights"],
            test_img_dir=args.test_img_dir,
            test_lbl_dir=args.test_lbl_dir,
            img_size=args.img_size,
            device=device,
            exp_name=args.exp_name,
        )
        csv_test = Path("results/metrics") / f"{args.exp_name}_test.csv"
        csv_test.parent.mkdir(parents=True, exist_ok=True)
        with csv_test.open("w", newline="") as f:
            keys = ["exp", "split", "map50", "map50_95", "precision", "recall", "f1"]
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerow(test_result)

        # copiar mejor checkpoint a models/yolo_baseline/best.pt
        best_ckpt = Path(best["weights"])
        dest = args.output_dir / "best.pt"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_ckpt, dest)
        logger.info(f"mejor checkpoint → {dest}")
        logger.info(f"resultado test → {csv_test}")


if __name__ == "__main__":
    main()
