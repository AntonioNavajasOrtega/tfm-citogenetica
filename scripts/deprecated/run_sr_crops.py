"""
run_sr_crops.py — Orquestador para aplicar ResShift a los recortes de cromosomas.

Itera sobre los subdirectorios de data/crops (ej. 0_normal, 1_dicentric)
y lanza el script de inferencia run_resshift.py sobre cada uno.

Uso:
    python scripts/diffusion/run_sr_crops.py --crops_dir data/crops --out_dir data/crops_sr --scale 2
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Aplicar ResShift a recortes de cromosomas")
    p.add_argument("--crops_dir", type=Path, default=Path("data/crops"), help="Directorio origen con recortes")
    p.add_argument("--out_dir", type=Path, default=Path("data/crops_sr"), help="Directorio destino para recortes SR")
    p.add_argument("--scale", type=int, default=2, help="Escala de super-resolución")
    p.add_argument("--version", type=str, default="v3", help="Versión de ResShift (v1, v2, v3)")
    return p.parse_args()


def main():
    args = parse_args()
    
    if not args.crops_dir.exists() or not any(args.crops_dir.iterdir()):
        print(f"[ERROR] Directorio de recortes vacio o no existe: {args.crops_dir}")
        sys.exit(1)
        
    print("=" * 60)
    print("SUPER-RESOLUCIÓN DE RECORTES (ResShift)")
    print("=" * 60)
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    run_resshift_script = Path("scripts/diffusion/run_resshift.py")
    if not run_resshift_script.exists():
        print(f"[ERROR] No se encuentra {run_resshift_script}")
        sys.exit(1)

    subdirs = [d for d in args.crops_dir.iterdir() if d.is_dir()]
    
    for subdir in subdirs:
        class_name = subdir.name
        print(f"\nProcesando clase: {class_name}")
        
        # Donde guardará ResShift internamente: args.out_dir / class_name
        # run_resshift.py normalmente guarda en output_dir/exp_name
        # Para mantener la misma estructura, lo llamamos de esta forma:
        
        cmd = [
            sys.executable, str(run_resshift_script),
            "--input_dir", str(subdir),
            "--output_dir", str(args.out_dir),
            "--scale", str(args.scale),
            "--version", args.version,
            "--exp_name", class_name  # Esto creará data/crops_sr/0_normal/ etc.
        ]
        
        try:
            print(f"Ejecutando: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] Falló ResShift en el directorio {subdir}")
            continue

    print("\n[INFO] Super-resolución de recortes completada.")
    print(f"[INFO] Resultados guardados en: {args.out_dir}")

if __name__ == "__main__":
    main()
