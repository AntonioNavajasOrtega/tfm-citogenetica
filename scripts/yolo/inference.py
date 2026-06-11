import argparse
from pathlib import Path
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics no esta instalado. Ejecuta: pip install ultralytics")
    sys.exit(1)

def parse_args():
    p = argparse.ArgumentParser(description="Aplica un modelo YOLO entrenado a una carpeta de imagenes")
    p.add_argument("--model", type=Path, required=True,
                   help="Ruta al modelo entrenado (ej: models/experimentos/yolo11n_img640/best.pt)")
    p.add_argument("--source", type=Path, required=True,
                   help="Carpeta con las imagenes a evaluar (ej: data/test/images o data/sr)")
    p.add_argument("--img_size", type=int, default=640,
                   help="Resolucion a la que se hara la inferencia (DEBE ser la misma con la que se entreno)")
    p.add_argument("--output_dir", type=Path, default=Path("results/predictions"),
                   help="Carpeta donde se guardaran las imagenes con las predicciones dibujadas")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Umbral de confianza para mostrar predicciones (def: 0.25)")
    return p.parse_args()

def main():
    args = parse_args()
    
    if not args.model.exists():
        print(f"[ERROR] No se encuentra el modelo en: {args.model}")
        sys.exit(1)
        
    if not args.source.exists() or not args.source.is_dir():
        print(f"[ERROR] No se encuentra la carpeta origen: {args.source}")
        sys.exit(1)
        
    print("\n" + "="*60)
    print(" INFERENCIA YOLO")
    print("="*60)
    print(f" Modelo   : {args.model}")
    print(f" Origen   : {args.source}")
    print(f" Resoluc. : {args.img_size}x{args.img_size}")
    print(f" Confianza: {args.conf}")
    print(f" Destino  : {args.output_dir}")
    print("="*60 + "\n")
    
    # Cargar modelo
    model = YOLO(str(args.model))
    
    # Ejecutar inferencia
    # save=True guarda las imagenes con cajas
    # save_txt=True guarda las coordenadas en txt
    results = model.predict(
        source=str(args.source),
        imgsz=args.img_size,
        conf=args.conf,
        project=str(args.output_dir.parent),
        name=args.output_dir.name,
        save=True,
        save_txt=True,
        exist_ok=True
    )
    
    print(f"\n[INFO] Inferencia completada. Revisa las imagenes generadas en {args.output_dir}")

if __name__ == "__main__":
    main()
