"""
report_metrics.py — Presenta métricas YOLO (train/val) de forma legible
                    y genera tablas LaTeX listas para copiar en la memoria.

Modos de uso
------------

1. Resumen de un experimento con múltiples folds (estructura estándar del TFM):
       python scripts/report_metrics.py folds \
           --exp_dir  runs/detect/runs/linea_base/yolo11n_img640

2. Comparativa entre varios experimentos (una fila por experimento):
       python scripts/report_metrics.py compare \
           --exp_dirs runs/detect/runs/linea_base/yolo11n_img640 \
                      runs/detect/runs/linea_base/yolo11s_img640

3. Detalle de un único run (un solo fold o un único entrenamiento):
       python scripts/report_metrics.py single \
           --run_dir  runs/detect/runs/linea_base/yolo11n_img640/run_fold1

Flags comunes
-------------
   --out_dir   dónde guardar los .tex generados (default: results/latex)
   --no_latex  solo imprime por consola, no escribe ficheros .tex
   --best_only en modo "folds", toma la mejor época (max mAP50) en vez del último epoch
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

# Columnas que nos interesan del results.csv de YOLO train
TRAIN_COLS = {
    "epoch":                    "Época",
    "metrics/precision(B)":     "Precisión",
    "metrics/recall(B)":        "Recall",
    "metrics/mAP50(B)":         "mAP@50",
    "metrics/mAP50-95(B)":      "mAP@50-95",
    "train/box_loss":           "Loss Caja (train)",
    "val/box_loss":             "Loss Caja (val)",
    "train/cls_loss":           "Loss Clase (train)",
    "val/cls_loss":             "Loss Clase (val)",
}

# Columnas de métricas "puras" (sin pérdidas)
METRIC_COLS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]

METRIC_LABELS = {
    "metrics/precision(B)":  "Precisión",
    "metrics/recall(B)":     "Recall",
    "metrics/mAP50(B)":      "mAP@0.5",
    "metrics/mAP50-95(B)":   "mAP@0.5:0.95",
}

SEP = "─" * 72


# ─────────────────────────────────────────────────────────────────────────────
# Lectura de datos
# ─────────────────────────────────────────────────────────────────────────────

def load_results_csv(run_dir: Path) -> pd.DataFrame | None:
    """Lee results.csv de un run. Devuelve None si no existe."""
    csv = run_dir / "results.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    return df


def best_epoch_row(df: pd.DataFrame) -> pd.Series:
    """Fila con el mAP@50 más alto (época de mejor checkpoint)."""
    return df.loc[df["metrics/mAP50(B)"].idxmax()]


def last_epoch_row(df: pd.DataFrame) -> pd.Series:
    """Última fila del CSV."""
    return df.iloc[-1]


def compute_f1(row: pd.Series) -> float:
    p = row.get("metrics/precision(B)", 0.0)
    r = row.get("metrics/recall(B)", 0.0)
    return 2 * p * r / (p + r + 1e-8)


def find_fold_dirs(exp_dir: Path) -> list[Path]:
    """Devuelve subdirectorios que contienen 'fold' en el nombre, ordenados."""
    return sorted(d for d in exp_dir.iterdir() if d.is_dir() and "fold" in d.name.lower())


def load_args_yaml(run_dir: Path) -> dict:
    """Lee args.yaml si existe y devuelve un dict básico."""
    yaml_path = run_dir / "args.yaml"
    if not yaml_path.exists():
        return {}
    try:
        import yaml  # PyYAML (incluido con ultralytics)
        with yaml_path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Formateo de consola
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"


def print_header(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def print_metric_row(label: str, value: float, extra: str = "") -> None:
    bar_len = int(value * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)
    extra_str = f"  {extra}" if extra else ""
    print(f"  {label:<22} {fmt(value)}  [{bar}]{extra_str}")


def print_single_run_summary(run_dir: Path, use_best: bool = True) -> dict | None:
    """Imprime resumen de un único run y devuelve dict de métricas."""
    df = load_results_csv(run_dir)
    if df is None:
        print(f"  [!] Sin results.csv en {run_dir}")
        return None

    row = best_epoch_row(df) if use_best else last_epoch_row(df)
    epoch_tag = f"época {int(row['epoch'])}" if use_best else "última época"
    args = load_args_yaml(run_dir)

    print_header(f"Run: {run_dir.name}  ({epoch_tag})")

    if args:
        model = args.get("model", "—")
        imgsz = args.get("imgsz", "—")
        epochs_run = int(df["epoch"].max())
        print(f"  Modelo: {model}   imgsz: {imgsz}   épocas entrenadas: {epochs_run}")
        print()

    f1 = compute_f1(row)
    metrics = {
        "Precisión":    row["metrics/precision(B)"],
        "Recall":       row["metrics/recall(B)"],
        "F1":           f1,
        "mAP@0.5":      row["metrics/mAP50(B)"],
        "mAP@0.5:0.95": row["metrics/mAP50-95(B)"],
    }
    for label, val in metrics.items():
        print_metric_row(label, val)

    print(f"\n  Loss (val)  →  box: {fmt(row['val/box_loss'])}  "
          f"cls: {fmt(row['val/cls_loss'])}")
    return {**{"run": run_dir.name, "época": int(row["epoch"])}, **metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Modo: folds (un experimento con N folds)
# ─────────────────────────────────────────────────────────────────────────────

def mode_folds(args: argparse.Namespace) -> list[dict]:
    exp_dir = Path(args.exp_dir)
    fold_dirs = find_fold_dirs(exp_dir)

    if not fold_dirs:
        print(f"[ERROR] No se encontraron subdirectorios 'fold*' en {exp_dir}")
        sys.exit(1)

    print_header(f"EXPERIMENTO: {exp_dir.name}  —  {len(fold_dirs)} folds")

    all_rows = []
    for fd in fold_dirs:
        row = print_single_run_summary(fd, use_best=args.best_only)
        if row:
            all_rows.append(row)

    if not all_rows:
        print("[ERROR] No se pudieron leer métricas.")
        sys.exit(1)

    # Resumen estadístico
    metric_keys = ["Precisión", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95"]
    print_header(f"RESUMEN 5-FOLD — {exp_dir.name}")
    print(f"  {'Métrica':<22} {'Media':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'─'*22} {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for k in metric_keys:
        vals = [r[k] for r in all_rows]
        print(
            f"  {k:<22} {np.mean(vals):>8.4f}  {np.std(vals):>8.4f}"
            f"  {np.min(vals):>8.4f}  {np.max(vals):>8.4f}"
        )
    print()
    return all_rows


# ─────────────────────────────────────────────────────────────────────────────
# Modo: compare (varios experimentos, una fila cada uno)
# ─────────────────────────────────────────────────────────────────────────────

def mode_compare(args: argparse.Namespace) -> list[dict]:
    exp_dirs = [Path(p) for p in args.exp_dirs]
    summary_rows = []

    for exp_dir in exp_dirs:
        fold_dirs = find_fold_dirs(exp_dir)
        rows = []
        if fold_dirs:
            for fd in fold_dirs:
                df = load_results_csv(fd)
                if df is None:
                    continue
                row = best_epoch_row(df) if args.best_only else last_epoch_row(df)
                rows.append(row)
        else:
            # Un único run sin estructura de folds
            df = load_results_csv(exp_dir)
            if df is not None:
                row = best_epoch_row(df) if args.best_only else last_epoch_row(df)
                rows.append(row)

        if not rows:
            print(f"  [!] Sin datos en {exp_dir}")
            continue

        def mean_metric(col):
            return float(np.mean([r[col] for r in rows]))

        p = mean_metric("metrics/precision(B)")
        r = mean_metric("metrics/recall(B)")
        summary_rows.append({
            "Experimento":   exp_dir.name,
            "Folds":         len(rows),
            "Precisión":     p,
            "Recall":        r,
            "F1":            2 * p * r / (p + r + 1e-8),
            "mAP@0.5":       mean_metric("metrics/mAP50(B)"),
            "mAP@0.5:0.95":  mean_metric("metrics/mAP50-95(B)"),
        })

    if not summary_rows:
        print("[ERROR] No se obtuvieron datos.")
        sys.exit(1)

    # Tabla consola
    cols = ["Experimento", "Folds", "Precisión", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95"]
    col_w = [max(len(c), max(len(str(r[c])) for r in summary_rows)) + 2 for c in cols]

    print_header("COMPARATIVA DE EXPERIMENTOS")
    header = "  " + "  ".join(c.ljust(col_w[i]) for i, c in enumerate(cols))
    print(header)
    print("  " + "  ".join("─" * col_w[i] for i in range(len(cols))))
    for r in summary_rows:
        row_str = "  " + "  ".join(
            (str(r[c]) if c in ("Experimento", "Folds") else fmt(r[c])).ljust(col_w[i])
            for i, c in enumerate(cols)
        )
        print(row_str)
    print()
    return summary_rows


# ─────────────────────────────────────────────────────────────────────────────
# Modo: single (un único run)
# ─────────────────────────────────────────────────────────────────────────────

def mode_single(args: argparse.Namespace) -> list[dict]:
    row = print_single_run_summary(Path(args.run_dir), use_best=args.best_only)
    return [row] if row else []


# ─────────────────────────────────────────────────────────────────────────────
# Generación LaTeX
# ─────────────────────────────────────────────────────────────────────────────

def escape_latex(s: str) -> str:
    replacements = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$"}
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s


def folds_to_latex(rows: list[dict], exp_name: str) -> str:
    """Tabla LaTeX: una fila por fold + fila de media ± std."""
    metric_keys = ["Precisión", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95"]
    col_spec = "l" + "r" * len(metric_keys)
    header_cols = " & ".join(["\\textbf{Fold}"] + [f"\\textbf{{{k}}}" for k in metric_keys])

    lines = [
        "% ── Tabla generada por report_metrics.py ──",
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{Resultados validación cruzada 5-fold — {escape_latex(exp_name)}}}",
        f"  \\label{{tab:{exp_name.lower().replace(' ', '_')}}}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    {header_cols} \\\\",
        "    \\midrule",
    ]

    for r in rows:
        fold_label = escape_latex(r.get("run", r.get("Experimento", "—")))
        vals = " & ".join(fmt(r[k]) for k in metric_keys)
        lines.append(f"    {fold_label} & {vals} \\\\")

    # Media ± std
    lines.append("    \\midrule")
    mean_vals, std_vals = [], []
    for k in metric_keys:
        vals_num = [r[k] for r in rows]
        mean_vals.append(np.mean(vals_num))
        std_vals.append(np.std(vals_num))

    mean_str = " & ".join(
        f"${fmt(m)} \\pm {fmt(s)}$" for m, s in zip(mean_vals, std_vals)
    )
    lines.append(f"    \\textbf{{Media}} & {mean_str} \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def compare_to_latex(rows: list[dict], caption: str = "Comparativa de experimentos") -> str:
    """Tabla LaTeX: una fila por experimento."""
    metric_keys = ["Precisión", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95"]
    col_spec = "l" + "c" + "r" * len(metric_keys)
    header_cols = " & ".join(
        ["\\textbf{Experimento}", "\\textbf{Folds}"]
        + [f"\\textbf{{{k}}}" for k in metric_keys]
    )

    lines = [
        "% ── Tabla generada por report_metrics.py ──",
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{escape_latex(caption)}}}",
        "  \\label{tab:comparativa}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        "    \\toprule",
        f"    {header_cols} \\\\",
        "    \\midrule",
    ]

    best_map = max(r["mAP@0.5"] for r in rows)
    for r in rows:
        exp = escape_latex(r["Experimento"])
        folds = str(r["Folds"])
        vals = " & ".join(
            (f"\\textbf{{{fmt(r[k])}}}" if k == "mAP@0.5" and r[k] == best_map else fmt(r[k]))
            for k in metric_keys
        )
        lines.append(f"    {exp} & {folds} & {vals} \\\\")

    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def single_to_latex(rows: list[dict], caption: str) -> str:
    """Tabla LaTeX para un único run."""
    if not rows:
        return ""
    r = rows[0]
    metric_keys = ["Precisión", "Recall", "F1", "mAP@0.5", "mAP@0.5:0.95"]
    lines = [
        "% ── Tabla generada por report_metrics.py ──",
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{escape_latex(caption)}}}",
        "  \\label{tab:single_run}",
        "  \\begin{tabular}{lr}",
        "    \\toprule",
        "    \\textbf{Métrica} & \\textbf{Valor} \\\\",
        "    \\midrule",
    ]
    for k in metric_keys:
        lines.append(f"    {k} & {fmt(r[k])} \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Presenta métricas YOLO de forma legible y genera tablas LaTeX."
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ── folds ──
    pf = sub.add_parser("folds", help="Resumen de un experimento con múltiples folds")
    pf.add_argument(
        "--exp_dir", type=str, required=True,
        help="Carpeta del experimento que contiene subdirectorios fold* "
             "(ej: runs/detect/runs/linea_base/yolo11n_img640)"
    )

    # ── compare ──
    pc = sub.add_parser("compare", help="Comparativa entre varios experimentos")
    pc.add_argument(
        "--exp_dirs", nargs="+", required=True,
        help="Lista de carpetas de experimentos a comparar"
    )
    pc.add_argument(
        "--caption", type=str, default="Comparativa de experimentos YOLO",
        help="Título de la tabla LaTeX"
    )

    # ── single ──
    ps = sub.add_parser("single", help="Detalle de un único run/fold")
    ps.add_argument(
        "--run_dir", type=str, required=True,
        help="Carpeta de un único run (ej: runs/.../yolo11n_img640/run_fold1)"
    )
    ps.add_argument(
        "--caption", type=str, default="Resultados de entrenamiento YOLO",
        help="Título de la tabla LaTeX"
    )

    # Flags comunes
    for sp in [pf, pc, ps]:
        sp.add_argument(
            "--out_dir", type=str, default="results/latex",
            help="Directorio donde guardar los .tex (default: results/latex)"
        )
        sp.add_argument(
            "--no_latex", action="store_true",
            help="No generar ficheros .tex, solo imprimir por consola"
        )
        sp.add_argument(
            "--best_only", action="store_true", default=True,
            help="Usar la época con mejor mAP@50 (default: True)"
        )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Ejecutar modo seleccionado
    if args.mode == "folds":
        rows = mode_folds(args)
        latex = folds_to_latex(rows, Path(args.exp_dir).name)
        out_name = f"{Path(args.exp_dir).name}_folds.tex"

    elif args.mode == "compare":
        rows = mode_compare(args)
        latex = compare_to_latex(rows, args.caption)
        out_name = "comparativa.tex"

    elif args.mode == "single":
        rows = mode_single(args)
        caption = getattr(args, "caption", "Resultados de entrenamiento YOLO")
        latex = single_to_latex(rows, caption)
        out_name = f"{Path(args.run_dir).name}.tex"

    # Mostrar LaTeX en consola
    print(SEP)
    print("  TABLA LaTeX  (copiar en la memoria)")
    print(SEP)
    print()
    print(latex)
    print()

    # Guardar en fichero
    if not args.no_latex:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / out_name
        out_path.write_text(latex, encoding="utf-8")
        print(f"  ✓ Tabla guardada en: {out_path.resolve()}")


if __name__ == "__main__":
    main()
