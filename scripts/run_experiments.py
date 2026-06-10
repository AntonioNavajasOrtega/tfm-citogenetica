import argparse
import subprocess
import sys
import time
from pathlib import Path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ejecuta una bateria de experimentos iterando sobre parametros de YOLO")
    p.add_argument("--model_sizes", nargs="+", default=["n", "s"], 
                   help="Lista de tamaños de modelo a probar (ej: n s m)")
    p.add_argument("--img_sizes", nargs="+", type=int, default=[640, 1024], 
                   help="Lista de tamaños de imagen a probar (ej: 640 1024 1280)")
    p.add_argument("--epochs", type=int, default=500, 
                   help="Numero de epocas maxima por fold")
    p.add_argument("--patience", type=int, default=30, 
                   help="Epocas sin mejora antes del early stop")
    p.add_argument("--batch_size", type=int, default=-1, 
                   help="Tamaño de lote (use -1 para AutoBatch)")
    p.add_argument("--out_base_dir", type=Path, default=Path("models/experimentos"), 
                   help="Directorio base donde se guardaran los resultados de los experimentos")
    return p.parse_args()

def main():
    args = parse_args()
    
    python_exe = sys.executable
    train_script = Path("scripts/train_yolo.py")
    
    if not train_script.exists():
        print(f"[ERROR] No se encuentra {train_script}. Ejecuta el script desde la raiz del proyecto.")
        sys.exit(1)

    total_experiments = len(args.model_sizes) * len(args.img_sizes)
    current_exp = 1

    print("\n" + "#"*60)
    print(f" INICIANDO BATERIA DE EXPERIMENTOS NOCTURNOS ")
    print(f" Total experimentos a ejecutar: {total_experiments}")
    print("#"*60 + "\n")

    start_time = time.time()

    for m_size in args.model_sizes:
        for i_size in args.img_sizes:
            exp_name = f"yolo11{m_size}_img{i_size}"
            output_dir = args.out_base_dir / exp_name
            
            print(f"\n" + "="*50)
            print(f" EXPERIMENTO [{current_exp}/{total_experiments}]")
            print(f" Modelo: yolo11{m_size}")
            print(f" Resolucion: {i_size}x{i_size}")
            print(f" Guardando en: {output_dir}")
            print("="*50 + "\n")
            
            cmd = [
                python_exe, str(train_script),
                "--model_size", m_size,
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
                "--batch_size", str(args.batch_size),
                "--img_size", str(i_size),
                "--output_dir", str(output_dir),
                "--exp_name", exp_name
            ]
            
            # Ejecutar el comando y esperar a que termine
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"\n[ERROR] El experimento con yolo11{m_size} a {i_size} fallo. Continuando con el siguiente...")
            
            current_exp += 1

    end_time = time.time()
    elapsed_hours = (end_time - start_time) / 3600
    
    print("\n" + "#"*60)
    print(" BATERIA DE EXPERIMENTOS FINALIZADA CON EXITO! ")
    print(f" Tiempo total estimado: {elapsed_hours:.2f} horas")
    print(f" Resultados guardados en: {args.out_base_dir}")
    print("#"*60 + "\n")

if __name__ == "__main__":
    main()
