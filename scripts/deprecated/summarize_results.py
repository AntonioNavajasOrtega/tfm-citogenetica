import argparse
import csv
from pathlib import Path
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Resume los resultados de todos los experimentos YOLO")
    p.add_argument("--metrics_dir", type=Path, default=Path("results/metrics"),
                   help="Directorio donde se encuentran los CSV de metricas")
    p.add_argument("--output_csv", type=Path, default=Path("results/global_summary.csv"),
                   help="Archivo CSV donde guardar el resumen global")
    return p.parse_args()

def main():
    args = parse_args()
    
    if not args.metrics_dir.exists():
        print(f"[ERROR] No existe el directorio de metricas: {args.metrics_dir}")
        sys.exit(1)

    # Buscar todos los archivos _cv.csv
    cv_files = list(args.metrics_dir.glob("*_cv.csv"))
    
    if not cv_files:
        print("[AVISO] No se encontraron archivos *_cv.csv para resumir.")
        sys.exit(0)

    summary_data = []

    for cv_file in cv_files:
        exp_name = cv_file.name.replace("_cv.csv", "")
        
        try:
            with open(cv_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Buscamos la fila de la media
                    if row.get("fold") == "media":
                        summary_data.append({
                            "Experimento": exp_name,
                            "mAP50": float(row.get("map50", 0)),
                            "mAP50-95": float(row.get("map50_95", 0)),
                            "Precision": float(row.get("precision", 0)),
                            "Recall": float(row.get("recall", 0)),
                            "F1-Score": float(row.get("f1", 0))
                        })
                        break
        except Exception as e:
            print(f"[ERROR] Al leer {cv_file}: {e}")

    # Ordenar por mAP50 descendente
    summary_data.sort(key=lambda x: x["mAP50"], reverse=True)

    print("\n" + "="*80)
    print(" RESUMEN GLOBAL DE EXPERIMENTOS (Basado en media de 5-Fold)")
    print("="*80)
    print(f"{'Experimento':<25} | {'mAP50':<8} | {'mAP50-95':<9} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8}")
    print("-" * 80)
    for d in summary_data:
        print(f"{d['Experimento']:<25} | {d['mAP50']:<8.4f} | {d['mAP50-95']:<9.4f} | {d['Precision']:<10.4f} | {d['Recall']:<8.4f} | {d['F1-Score']:<8.4f}")
    print("="*80 + "\n")

    # Guardar a CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        keys = ["Experimento", "mAP50", "mAP50-95", "Precision", "Recall", "F1-Score"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summary_data)
        
    print(f"Resumen guardado correctamente en: {args.output_csv}\n")

if __name__ == "__main__":
    main()
