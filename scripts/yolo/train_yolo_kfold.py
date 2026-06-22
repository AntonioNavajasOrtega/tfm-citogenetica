"""
train_yolo_kfold.py — Entrena YOLO en los folds pre-generados y recoge métricas.

Requiere haber ejecutado generate_folds.py primero.
Lee directamente los dataset.yaml de cada fold en --folds_dir.

Tras cada fold extrae Precisión, Recall, mAP@0.5 y mAP@0.5-95 de la
mejor época y guarda un JSON de resumen en results/yolo_kfold_metrics_{model}.json.

Uso:
    python scripts/yolo/train_yolo_kfold.py --model yolo11s.pt --epochs 100 --patience 20
"""

import argparse
import gc
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import torch

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("Instala pandas: pip install pandas")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("Instala ultralytics: pip install ultralytics")
    sys.exit(1)


# ── Columnas de métricas de interés en results.csv de ultralytics ─────────────
METRIC_COLS = {
    "metrics/precision(B)":  "precision",
    "metrics/recall(B)":     "recall",
    "metrics/mAP50(B)":      "mAP50",
    "metrics/mAP50-95(B)":   "mAP50_95",
}


# ── Parche de torch.save con escritura atómica + retry ───────────────────────
def patch_torch_save_with_retry(max_retries: int = 5) -> None:
    """
    Reemplaza torch.save con una versión que:
      1. Escribe primero a un archivo .tmp
      2. Lo mueve atómicamente al destino final (evita archivos corruptos)
      3. Reintenta con back-off exponencial ante RuntimeError / ValueError / OSError

    Esto resuelve:
      - ValueError: I/O operation on closed file  (torch nightly + CUDA en Windows)
      - RuntimeError: [WinError 32] archivo en uso  (antivirus)
    """
    _wrapped_save = torch.save

    def _save_with_retry(*args, **kwargs):
        dest = args[1] if len(args) > 1 else kwargs.get("f")

        if isinstance(dest, (str, Path)):
            dest = Path(dest)
            tmp_path = dest.with_suffix(".tmp")

            for attempt in range(max_retries):
                try:
                    _wrapped_save(args[0], tmp_path, *args[2:], **kwargs)
                    tmp_path.replace(dest)   # movimiento atómico
                    return
                except (RuntimeError, ValueError, OSError) as exc:
                    if tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)
                    if attempt == max_retries - 1:
                        raise
                    wait = 0.5 * (2 ** attempt)   # 0.5 s, 1 s, 2 s, 4 s, …
                    logging.getLogger(__name__).warning(
                        f"torch.save fallo ({type(exc).__name__}: {exc}) — "
                        f"reintento {attempt + 1}/{max_retries - 1} en {wait:.1f}s"
                    )
                    time.sleep(wait)
        else:
            # File handle directo: no podemos interceptar, intentamos tal cual
            return _wrapped_save(*args, **kwargs)

    torch.save = _save_with_retry


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Entrena YOLO en K-Fold con folds pre-generados.")
    p.add_argument("--folds_dir",  type=Path, default=Path("data/yolo_kfold"),
                   help="Directorio raíz que contiene fold_1/, fold_2/, …")
    p.add_argument("--model",      type=str,  default="yolo11n.pt",
                   help="Checkpoint de partida (ej. yolo11n.pt, yolo11s.pt, yolo11m.pt).")
    p.add_argument("--epochs",     type=int,  default=100)
    p.add_argument("--patience",   type=int,  default=30,
                   help="Early stopping: épocas sin mejora antes de parar.")
    p.add_argument("--img_size",   type=int,  default=1280)
    p.add_argument("--batch_size", type=int,  default=-1)
    p.add_argument("--seed",       type=int,  default=42)
    p.add_argument("--device",     default="0", help="Dispositivo CUDA (ej. 0) o 'cpu'.")
    p.add_argument("--workers",    type=int,  default=2,
                   help="Workers del dataloader. Reducir a 2 evita OOM de RAM en Windows "
                        "con múltiples folds. (por defecto: 2)")
    p.add_argument("--metrics_out", type=Path, default=None,
                   help="Ruta del JSON de resumen de métricas. "
                        "Por defecto: results/yolo_kfold_metrics_{model_stem}.json")
    p.add_argument("--folds",      type=int,  nargs="+", default=None,
                   help="Entrenar solo estos índices de fold (ej. --folds 1 3). "
                        "Por defecto entrena todos.")
    return p.parse_args()


def extract_best_metrics(results_csv: Path) -> dict | None:
    """Lee results.csv y devuelve las métricas de la mejor época (max mAP50)."""
    if not results_csv.exists():
        return None
    try:
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()
        best_row = df.loc[df["metrics/mAP50(B)"].idxmax()]
        metrics = {"best_epoch": int(best_row["epoch"]), "total_epochs": int(df["epoch"].max())}
        for csv_col, key in METRIC_COLS.items():
            metrics[key] = round(float(best_row[csv_col]), 6)
        p, r = metrics["precision"], metrics["recall"]
        metrics["f1"] = round(2 * p * r / (p + r + 1e-8), 6)
        return metrics
    except Exception as e:
        logging.getLogger(__name__).warning(f"No se pudieron leer métricas de {results_csv}: {e}")
        return None


def print_metrics_table(fold_metrics: dict, model_name: str, logger) -> None:
    """Imprime resumen de métricas por fold y medias globales."""
    keys = ["precision", "recall", "f1", "mAP50", "mAP50_95"]
    sep = "─" * 72
    logger.info(sep)
    logger.info(f"  MÉTRICAS K-FOLD — {model_name}")
    logger.info(sep)
    header = f"  {'Fold':<10} {'Prec':>8}  {'Recall':>8}  {'F1':>8}  {'mAP50':>8}  {'mAP50-95':>10}"
    logger.info(header)
    logger.info(f"  {'─'*10} {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}")

    collected = {k: [] for k in keys}
    for fold_name, m in sorted(fold_metrics.items()):
        row = f"  {fold_name:<10}"
        for k in keys:
            v = m.get(k, float("nan"))
            collected[k].append(v)
            row += f" {v:>8.4f} "
        logger.info(row)

    logger.info(f"  {'─'*10} {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}")
    mean_row = f"  {'Media':<10}"
    std_row  = f"  {'Std':<10}"
    for k in keys:
        vals = [v for v in collected[k] if not np.isnan(v)]
        mean_row += f" {np.mean(vals):>8.4f} " if vals else f" {'—':>8} "
        std_row  += f" {np.std(vals):>8.4f} "  if vals else f" {'—':>8} "
    logger.info(mean_row)
    logger.info(std_row)
    logger.info(sep)


def main():
    args = parse_args()
    logger = setup_logging()

    model_stem = Path(args.model).stem  # e.g. "yolo11s"

    if args.metrics_out is None:
        args.metrics_out = Path(f"results/yolo_kfold_metrics_{model_stem}.json")

    # ── Descubrir folds disponibles ────────────────────────────────────────────
    if not args.folds_dir.exists():
        logger.error(
            f"El directorio de folds no existe: {args.folds_dir}\n"
            "Ejecuta generate_folds.py primero."
        )
        sys.exit(1)

    fold_dirs = sorted(
        d for d in args.folds_dir.iterdir()
        if d.is_dir() and d.name.startswith("fold_")
    )
    if not fold_dirs:
        logger.error(f"No se encontraron carpetas fold_* en {args.folds_dir}.")
        sys.exit(1)

    if args.folds is not None:
        requested = {f"fold_{i}" for i in args.folds}
        fold_dirs = [d for d in fold_dirs if d.name in requested]
        if not fold_dirs:
            logger.error(f"Ninguno de los folds solicitados {args.folds} existe en {args.folds_dir}.")
            sys.exit(1)

    logger.info(f"Modelo:       {args.model}")
    logger.info(f"Folds:        {[d.name for d in fold_dirs]}")
    logger.info(f"Epochs:       {args.epochs}  |  Patience: {args.patience}")
    logger.info(f"Img size:     {args.img_size}  |  Batch: {args.batch_size}")
    logger.info(f"Workers:      {args.workers}")

    # ── Parche torch.save antes de entrenar ────────────────────────────────────
    patch_torch_save_with_retry()
    logger.info("torch.save parchado con escritura atómica + retry (max 5 intentos)")

    # ── Entrenar cada fold ─────────────────────────────────────────────────────
    fold_metrics: dict[str, dict] = {}

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        yaml_path = fold_dir / "dataset.yaml"

        if not yaml_path.exists():
            logger.error(
                f"No se encontró {yaml_path}.\n"
                "Vuelve a ejecutar generate_folds.py para regenerar los folds."
            )
            continue

        logger.info("=" * 60)
        logger.info(f"ENTRENANDO YOLO ({model_stem}) — {fold_name.upper()}")
        logger.info("=" * 60)

        model = YOLO(args.model)
        project_dir = (Path.cwd() / "models" / "yolo_kfold" / model_stem / fold_name).resolve()
        train_dir   = project_dir / "train"

        if train_dir.exists():
            logger.info(f"[{fold_name}] Limpiando run previo en {train_dir}")
            shutil.rmtree(train_dir)
        (train_dir / "weights").mkdir(parents=True, exist_ok=True)

        model.train(
            data=str(yaml_path.resolve()),
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.img_size,
            batch=args.batch_size,
            device=args.device,
            project=str(project_dir),
            name="train",
            exist_ok=True,
            seed=args.seed,
            verbose=False,
            plots=True,
            workers=args.workers,   # reducido para evitar OOM de RAM en Windows
        )

        # ── Liberar VRAM y RAM antes del siguiente fold ────────────────────────
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info(
                f"[{fold_name}] VRAM liberada: "
                f"{torch.cuda.memory_allocated() / 1e9:.2f} GB en uso"
            )
        # Pausa para que el SO recupere la RAM de los workers muertos
        logger.info(f"[{fold_name}] Esperando 5 s para liberar RAM de workers...")
        time.sleep(5)

        # ── Extraer métricas de la mejor época ────────────────────────────────
        results_csv = project_dir / "train" / "results.csv"
        metrics = extract_best_metrics(results_csv)
        if metrics:
            fold_metrics[fold_name] = metrics
            logger.info(
                f"[{fold_name}] Mejor época {metrics['best_epoch']}/{metrics['total_epochs']} → "
                f"mAP50={metrics['mAP50']:.4f}  "
                f"Prec={metrics['precision']:.4f}  "
                f"Recall={metrics['recall']:.4f}  "
                f"F1={metrics['f1']:.4f}"
            )
        else:
            logger.warning(f"[{fold_name}] No se pudieron leer métricas.")

    # ── Resumen final ──────────────────────────────────────────────────────────
    if fold_metrics:
        print_metrics_table(fold_metrics, model_stem, logger)

        keys = ["precision", "recall", "f1", "mAP50", "mAP50_95"]
        summary = {
            "model":  model_stem,
            "folds":  fold_metrics,
            "mean":   {},
            "std":    {},
        }
        for k in keys:
            vals = [m[k] for m in fold_metrics.values() if k in m]
            summary["mean"][k] = round(float(np.mean(vals)), 6) if vals else None
            summary["std"][k]  = round(float(np.std(vals)),  6) if vals else None

        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(f"✓ Métricas guardadas en: {args.metrics_out}")

    logger.info("Entrenamiento K-Fold YOLO completado.")


if __name__ == "__main__":
    main()