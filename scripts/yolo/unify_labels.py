import argparse
from pathlib import Path
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Convierte todas las etiquetas YOLO a clase 0 (cromosoma)")
    p.add_argument("--input_dir", type=Path, default=Path("data/raw/labels"), help="Directorio con etiquetas originales (2 clases)")
    p.add_argument("--output_dir", type=Path, default=Path("data/raw/labels_1class"), help="Directorio donde guardar etiquetas unificadas (1 clase)")
    return p.parse_args()

def main():
    args = parse_args()
    
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        print(f"[ERROR] No se encuentra el directorio de entrada: {args.input_dir}")
        sys.exit(1)
        
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    txt_files = list(args.input_dir.glob("*.txt"))
    if not txt_files:
        print(f"[WARNING] No se encontraron archivos .txt en {args.input_dir}")
        sys.exit(0)
        
    processed = 0
    for txt_path in txt_files:
        lines = txt_path.read_text().splitlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            # parts[0] es la clase, del 1 al 4 son los bboxes
            # Forzamos la clase a 0
            parts[0] = "0"
            new_lines.append(" ".join(parts))
            
        out_path = args.output_dir / txt_path.name
        out_path.write_text("\n".join(new_lines) + "\n")
        processed += 1
        
    print(f"[INFO] Procesados {processed} archivos.")
    print(f"[INFO] Etiquetas unificadas guardadas en {args.output_dir}")

if __name__ == "__main__":
    main()
