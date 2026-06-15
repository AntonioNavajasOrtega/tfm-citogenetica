"""
compare_results.py — Tabla comparativa de todos los experimentos SR.

Lee todos los CSV de métricas de imagen (PSNR/SSIM/LPIPS) y el CSV de YOLO,
y genera:
  - results/metrics/summary_sr.csv     — tabla completa por experimento
  - results/metrics/summary_sr.md      — versión markdown para el TFM

Uso:
    python scripts/diffusion/compare_results.py
    python scripts/diffusion/compare_results.py --metrics_dir results/metrics
"""

import argparse
import csv
import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tabla comparativa de experimentos SR")
    p.add_argument("--metrics_dir", type=Path, default=Path("results/metrics"),
                   help="Directorio con los CSV de métricas individuales")
    p.add_argument("--yolo_csv", type=Path, default=None,
                   help="CSV de resultados YOLO (por defecto: metrics_dir/yolo_comparison.csv)")
    p.add_argument("--output_csv", type=Path, default=None,
                   help="CSV de salida (por defecto: metrics_dir/summary_sr.csv)")
    p.add_argument("--output_md",  type=Path, default=None,
                   help="Markdown de salida (por defecto: metrics_dir/summary_sr.md)")
    p.add_argument("--log_level",  default="INFO")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Lectura de métricas de imagen
# ─────────────────────────────────────────────────────────────────────────────

def read_image_metrics(metrics_dir: Path) -> dict[str, dict]:
    """
    Lee todos los CSV individuales (uno por experimento SR) y extrae
    las filas de media (media_psnr, media_ssim, media_lpips).
    Devuelve {exp_name: {"psnr": float, "ssim": float, "lpips": float}}.
    """
    results = {}
    # Excluir los CSVs de resumen y de YOLO
    skip = {"summary_sr.csv", "yolo_comparison.csv", "metrics.csv"}

    for csv_path in sorted(metrics_dir.glob("*.csv")):
        if csv_path.name in skip:
            continue
        exp_name = csv_path.stem
        row_data: dict[str, float] = {}

        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row.get("image", "")
                for metric in ("psnr", "ssim", "lpips"):
                    if img == f"media_{metric}" and metric in row:
                        try:
                            row_data[metric] = float(row[metric])
                        except (ValueError, KeyError):
                            pass

        if row_data:
            results[exp_name] = row_data

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Lectura de resultados YOLO
# ─────────────────────────────────────────────────────────────────────────────

def read_yolo_metrics(yolo_csv: Path) -> dict[str, dict]:
    """
    Lee el CSV de comparación YOLO y devuelve
    {exp_name: {"map50": float, "map50_95": float, "precision": float,
                "recall": float, "f1": float}}.
    """
    results = {}
    if not yolo_csv.exists():
        return results

    with yolo_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp = row.get("exp", "").strip()
            if not exp:
                continue
            try:
                results[exp] = {
                    "map50":     float(row.get("map50",    0)),
                    "map50_95":  float(row.get("map50_95", 0)),
                    "precision": float(row.get("precision", 0)),
                    "recall":    float(row.get("recall",   0)),
                    "f1":        float(row.get("f1",       0)),
                }
            except (ValueError, KeyError):
                pass

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de tabla unificada
# ─────────────────────────────────────────────────────────────────────────────

# Orden deseado de experimentos en la tabla
_EXP_ORDER = [
    "baseline_no_sr",
    "stablesr_x2_recon",
    "stablesr_x4_recon",
    "stablesr_x2_upscale",
    "resshift_pretrained_x2_recon",
    "resshift_pretrained_x4_recon",
    "resshift_pretrained_x4_upscale",
    "resshift_finetuned_x2_recon",
    "resshift_finetuned_x4_recon",
]

_IMG_METRICS  = ["psnr", "ssim", "lpips"]
_YOLO_METRICS = ["map50", "map50_95", "precision", "recall", "f1"]
_ALL_COLS     = ["exp"] + _IMG_METRICS + _YOLO_METRICS


def build_table(img_metrics: dict, yolo_metrics: dict) -> list[dict]:
    """
    Une métricas de imagen y YOLO en una tabla unificada.
    Para experimentos de upscale (sin ref HR) las columnas de imagen quedan vacías.
    """
    all_exp_names = sorted(
        set(img_metrics) | set(yolo_metrics),
        key=lambda e: (_EXP_ORDER.index(e) if e in _EXP_ORDER else 999, e),
    )

    rows = []
    for exp in all_exp_names:
        row = {"exp": exp}
        for m in _IMG_METRICS:
            row[m] = img_metrics.get(exp, {}).get(m, "")
        for m in _YOLO_METRICS:
            # Para experimentos YOLO el nombre puede incluir "_yolo" al final
            yolo_key = exp if exp in yolo_metrics else f"{exp}_yolo"
            row[m] = yolo_metrics.get(yolo_key, {}).get(m, "")
        rows.append(row)

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Formateo Markdown
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val, decimals: int = 4) -> str:
    if val == "" or val is None:
        return "—"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def table_to_markdown(rows: list[dict]) -> str:
    """Genera una tabla Markdown con las columnas de la tabla comparativa."""
    header = (
        "| Experimento | PSNR ↑ | SSIM ↑ | LPIPS ↓ "
        "| mAP@50 ↑ | mAP@50-95 ↑ | Precision ↑ | Recall ↑ | F1 ↑ |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        exp  = r["exp"]
        psnr = _fmt(r.get("psnr"))
        ssim = _fmt(r.get("ssim"))
        lpips= _fmt(r.get("lpips"))
        m50  = _fmt(r.get("map50"))
        m595 = _fmt(r.get("map50_95"))
        prec = _fmt(r.get("precision"))
        rec  = _fmt(r.get("recall"))
        f1   = _fmt(r.get("f1"))
        lines.append(
            f"| {exp} | {psnr} | {ssim} | {lpips} "
            f"| {m50} | {m595} | {prec} | {rec} | {f1} |"
        )
    return "\n".join(lines)


def best_highlight(rows: list[dict]) -> str:
    """Genera un pequeño resumen de los mejores valores por métrica."""
    lines = ["\n## Mejores valores por métrica\n"]
    for metric, higher_is_better in [
        ("psnr", True), ("ssim", True), ("lpips", False),
        ("map50", True), ("map50_95", True), ("f1", True),
    ]:
        vals = [(r["exp"], r.get(metric)) for r in rows if r.get(metric) != ""]
        vals = [(e, float(v)) for e, v in vals if v != ""]
        if not vals:
            continue
        best_exp, best_val = (max if higher_is_better else min)(vals, key=lambda x: x[1])
        arrow = "↑" if higher_is_better else "↓"
        lines.append(f"- **{metric.upper()} {arrow}**: `{best_exp}` → {best_val:.4f}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)

    metrics_dir = args.metrics_dir
    yolo_csv    = args.yolo_csv  or metrics_dir / "yolo_comparison.csv"
    output_csv  = args.output_csv or metrics_dir / "summary_sr.csv"
    output_md   = args.output_md  or metrics_dir / "summary_sr.md"

    logger.info("=" * 60)
    logger.info("TABLA COMPARATIVA DE EXPERIMENTOS SR")
    logger.info("=" * 60)
    logger.info(f"  métricas dir : {metrics_dir}")
    logger.info(f"  yolo csv     : {yolo_csv}")
    logger.info(f"  output csv   : {output_csv}")
    logger.info(f"  output md    : {output_md}")

    if not metrics_dir.exists():
        logger.error(f"Directorio de métricas no encontrado: {metrics_dir}")
        sys.exit(1)

    # Leer métricas
    img_metrics  = read_image_metrics(metrics_dir)
    yolo_metrics = read_yolo_metrics(yolo_csv)

    logger.info(f"Experimentos con métricas de imagen: {list(img_metrics)}")
    logger.info(f"Experimentos con métricas YOLO:      {list(yolo_metrics)}")

    if not img_metrics and not yolo_metrics:
        logger.warning("No se encontraron métricas. Ejecuta primero las fases SR, métricas y YOLO.")
        sys.exit(0)

    rows = build_table(img_metrics, yolo_metrics)

    # ── CSV unificado ─────────────────────────────────────────────────────────
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_ALL_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"\n✓ CSV guardado en: {output_csv}")

    # ── Markdown ──────────────────────────────────────────────────────────────
    md_content = (
        "# Resultados comparativos — Superresolución\n\n"
        + table_to_markdown(rows)
        + "\n"
        + best_highlight(rows)
        + "\n\n---\n*Generado automáticamente por `compare_results.py`*\n"
    )
    output_md.write_text(md_content, encoding="utf-8")
    logger.info(f"✓ Markdown guardado en: {output_md}")

    # ── Resumen en consola ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN")
    logger.info("=" * 60)
    col_w = max(len(r["exp"]) for r in rows) + 2
    header = f"{'Experimento':<{col_w}} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'mAP50':>8} {'F1':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for r in rows:
        logger.info(
            f"{r['exp']:<{col_w}}"
            f" {_fmt(r.get('psnr')):>8}"
            f" {_fmt(r.get('ssim')):>8}"
            f" {_fmt(r.get('lpips')):>8}"
            f" {_fmt(r.get('map50')):>8}"
            f" {_fmt(r.get('f1')):>8}"
        )


if __name__ == "__main__":
    main()
