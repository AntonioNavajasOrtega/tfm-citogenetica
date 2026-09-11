"""
predict_folds.py — evaluación YOLO (val) sobre el conjunto de test usando
                   el best.pt de cada fold entrenado.

Calcula mAP@0.5, mAP@0.5:0.95, Precisión, Recall y F1 por fold,
y genera un resumen con media y desviación estándar al final.
Los resultados se guardan en un CSV.

Estructura esperada de --models_dir:
    <models_dir>/
        <exp_name>_fold1/train/weights/best.pt
        <exp_name>_fold2/train/weights/best.pt
        ...

  o bien (sin prefijo):
        fold1/train/weights/best.pt  |  fold1/weights/best.pt  |  fold1/best.pt
        ...

El YAML de validación debe existir en la carpeta de test (--yaml):
    data/test/test.yaml

Uso básico:
    python scripts/predict_folds.py \
        --yaml        data/test/test.yaml \
        --models_dir  runs/detect/runs/linea_base/yolo11n_img640

Opciones:
    --img_size   tamaño de inferencia   (default: 640)
    --device     "cuda" | "cpu"         (default: cuda si disponible)
    --exp_name   prefijo de carpeta de fold (ej: "run" → run_fold1…)
    --output_dir carpeta de salida      (default: results/predict/<models_dir.name>)
    --conf       umbral de confianza    (default: 0.25)
    --iou        umbral IoU NMS         (default: 0.45)
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("Instala ultralytics: pip install ultralytics")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def check_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            logging.getLogger(__name__).info(
                f"GPU detectada: {torch.cuda.get_device_name(0)}"
            )
            return "0"
        logging.getLogger(__name__).warning("CUDA no disponible — usando CPU")
        return "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Búsqueda de pesos por fold
# ---------------------------------------------------------------------------

def find_fold_weights(models_dir: Path, exp_name: str | None) -> list[tuple[str, Path]]:
    """
    Devuelve lista ordenada de (nombre_fold, ruta_best.pt).
    Busca best.pt en: train/weights/best.pt → weights/best.pt → best.pt
    """
    logger = logging.getLogger(__name__)

    def locate_best(fold_dir: Path) -> Path | None:
        for candidate in [
            fold_dir / "train" / "weights" / "best.pt",
            fold_dir / "weights" / "best.pt",
            fold_dir / "best.pt",
        ]:
            if candidate.exists():
                return candidate
        return None

    subdirs = sorted(d for d in models_dir.iterdir() if d.is_dir())

    if exp_name:
        prefix = f"{exp_name}_fold"
        filtered = [d for d in subdirs if d.name.startswith(prefix)]
        if not filtered:
            logger.warning(
                f"Sin subdirectorios con prefijo '{prefix}'. "
                "Buscando cualquier carpeta con 'fold'."
            )
            filtered = [d for d in subdirs if "fold" in d.name.lower()]
        subdirs = filtered
    else:
        subdirs = [d for d in subdirs if "fold" in d.name.lower()]

    results = []
    for fold_dir in subdirs:
        w = locate_best(fold_dir)
        if w is None:
            logger.warning(f"  [{fold_dir.name}] No se encontró best.pt — omitiendo")
            continue
        logger.info(f"  [{fold_dir.name}] pesos: {w}")
        results.append((fold_dir.name, w))

    return results


# ---------------------------------------------------------------------------
# Evaluación por fold
# ---------------------------------------------------------------------------

def evaluate_fold(
    fold_name: str,
    weights: Path,
    yaml_path: Path,
    img_size: int,
    device: str,
    conf: float,
    iou: float,
    output_dir: Path,
) -> dict:
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD : {fold_name}")
    logger.info(f"  pesos  : {weights}")
    logger.info(f"  yaml   : {yaml_path}")
    logger.info(f"  conf={conf}  iou={iou}  imgsz={img_size}  device={device}")
    logger.info(f"{'='*60}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(yaml_path.resolve()),
        imgsz=img_size,
        device=device,
        conf=conf,
        iou=iou,
        project=str(output_dir),
        name=fold_name,
        exist_ok=True,
        verbose=True,
        plots=True,
    )

    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    result = {
        "fold":      fold_name,
        "map50":     float(metrics.box.map50),
        "map50_95":  float(metrics.box.map),
        "precision": mp,
        "recall":    mr,
        "f1":        2 * mp * mr / (mp + mr + 1e-8),
    }

    logger.info(
        f"  map50={result['map50']:.4f}  map50-95={result['map50_95']:.4f}  "
        f"prec={result['precision']:.4f}  rec={result['recall']:.4f}  "
        f"f1={result['f1']:.4f}"
    )
    return result


# ---------------------------------------------------------------------------
# Resumen y CSV
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], csv_path: Path) -> None:
    logger = logging.getLogger(__name__)
    keys = ["map50", "map50_95", "precision", "recall", "f1"]

    logger.info("\n" + "=" * 70)
    logger.info("RESUMEN POR FOLD")
    logger.info("=" * 70)
    logger.info(f"{'fold':<20} {'map50':>8} {'map50-95':>10} {'prec':>8} {'rec':>8} {'f1':>8}")
    logger.info("-" * 70)
    for r in results:
        logger.info(
            f"{r['fold']:<20} {r['map50']:>8.4f} {r['map50_95']:>10.4f} "
            f"{r['precision']:>8.4f} {r['recall']:>8.4f} {r['f1']:>8.4f}"
        )

    means = {k: float(np.mean([r[k] for r in results])) for k in keys}
    stds  = {k: float(np.std( [r[k] for r in results])) for k in keys}

    logger.info("-" * 70)
    logger.info(
        f"{'MEDIA':<20} {means['map50']:>8.4f} {means['map50_95']:>10.4f} "
        f"{means['precision']:>8.4f} {means['recall']:>8.4f} {means['f1']:>8.4f}"
    )
    logger.info(
        f"{'STD':<20} {stds['map50']:>8.4f} {stds['map50_95']:>10.4f} "
        f"{stds['precision']:>8.4f} {stds['recall']:>8.4f} {stds['f1']:>8.4f}"
    )
    logger.info("=" * 70)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold"] + keys)
        writer.writeheader()
        writer.writerows([{k: r[k] for k in ["fold"] + keys} for r in results])
        writer.writerow({"fold": "MEDIA", **means})
        writer.writerow({"fold": "STD",   **stds})

    logger.info(f"Métricas guardadas en: {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluación YOLO multi-fold: mAP, precisión, recall y F1 "
                    "sobre el conjunto de test, fold a fold."
    )
    p.add_argument(
        "--yaml", type=Path, default=Path("data/test/test.yaml"),
        help="Ruta al YAML del conjunto de test (default: data/test/test.yaml)"
    )
    p.add_argument(
        "--models_dir", type=Path, required=True,
        help="Carpeta que contiene los subdirectorios de cada fold "
             "(ej: runs/detect/runs/linea_base/yolo11n_img640)"
    )
    p.add_argument(
        "--output_dir", type=Path, default=None,
        help="Dónde guardar resultados. Por defecto: results/predict/<nombre_models_dir>"
    )
    p.add_argument(
        "--exp_name", type=str, default=None,
        help="Prefijo de carpeta de fold (ej: 'run' → run_fold1…). "
             "Si se omite se detecta cualquier carpeta con 'fold'."
    )
    p.add_argument("--conf",     type=float, default=0.25, help="Umbral de confianza (default: 0.25)")
    p.add_argument("--iou",      type=float, default=0.45, help="Umbral IoU para NMS (default: 0.45)")
    p.add_argument("--img_size", type=int,   default=640,  help="Tamaño de imagen (default: 640)")
    p.add_argument("--device",   type=str,   default="cuda",
                   choices=["cuda", "cpu"], help="Dispositivo (default: cuda)")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)
    device = check_device(args.device)

    # Validar YAML
    if not args.yaml.exists():
        logger.error(
            f"No se encontró el YAML de test: {args.yaml}\n"
            f"Comprueba la ruta o créalo en data/test/test.yaml"
        )
        sys.exit(1)
    logger.info(f"YAML de test: {args.yaml.resolve()}")

    # Validar carpeta de modelos
    if not args.models_dir.exists():
        logger.error(f"No existe la carpeta de modelos: {args.models_dir}")
        sys.exit(1)

    # Directorio de salida
    output_dir = args.output_dir or Path("results/predict") / args.models_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorio de salida: {output_dir.resolve()}")

    # Buscar pesos de cada fold
    logger.info(f"\nBuscando folds en: {args.models_dir}")
    fold_weights = find_fold_weights(args.models_dir, args.exp_name)
    if not fold_weights:
        logger.error(
            "No se encontró ningún best.pt. Comprueba --models_dir y --exp_name.\n"
            "Estructura esperada: <fold_dir>/train/weights/best.pt"
        )
        sys.exit(1)
    logger.info(f"Folds encontrados: {len(fold_weights)}\n")

    # Evaluar cada fold
    all_results = []
    for fold_name, weights_path in fold_weights:
        result = evaluate_fold(
            fold_name=fold_name,
            weights=weights_path,
            yaml_path=args.yaml,
            img_size=args.img_size,
            device=device,
            conf=args.conf,
            iou=args.iou,
            output_dir=output_dir,
        )
        all_results.append(result)

    # Resumen + CSV
    csv_path = output_dir / f"{args.models_dir.name}_metrics.csv"
    print_summary(all_results, csv_path)


if __name__ == "__main__":
    main()
