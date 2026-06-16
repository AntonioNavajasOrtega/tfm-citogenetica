"""
crop_chromosomes.py — realiza inferencia YOLO, recorta cromosomas y asigna etiquetas.

1. Preprocesa la imagen (elimina artefactos).
2. Pasa YOLO (1 clase) para detectar cromosomas.
3. Calcula el IoU de cada predicción con el Ground Truth original (2 clases).
4. Asigna la clase (0=normal, 1=dicéntrico) al recorte y lo guarda en carpetas separadas.

Uso:
    python scripts/yolo/crop_chromosomes.py --model models/yolo_1class/yolo_1class/train/weights/best.pt
"""

import argparse
import cv2
import sys
from pathlib import Path
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("instala ultralytics: pip install ultralytics")
    sys.exit(1)

# Importamos la funcion de preprocesamiento existente
sys.path.append(str(Path(__file__).resolve().parents[2]))
try:
    from scripts.deprecated.preprocess import remove_circular_artifacts
except ImportError:
    print("Advertencia: No se pudo importar remove_circular_artifacts, se omitira el preprocesado.")
    remove_circular_artifacts = lambda x: x

def calculate_iou(box1, box2):
    """Calcula la Interseccion sobre Union (IoU) de dos cajas [x_center, y_center, w, h] normalizadas."""
    # Convertir a [x_min, y_min, x_max, y_max]
    b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    
    b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    
    # Interseccion
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
        
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    return inter_area / (b1_area + b2_area - inter_area)

def load_gt_labels(label_path):
    """Carga etiquetas GT y devuelve lista de tuplas (clase, [x, y, w, h])."""
    if not label_path.exists():
        return []
    gt_boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls = int(parts[0])
            box = [float(p) for p in parts[1:]]
            gt_boxes.append((cls, box))
    return gt_boxes

def parse_args():
    p = argparse.ArgumentParser(description="Crop chromosomes using YOLO and label via GT IoU")
    p.add_argument("--model", type=Path, required=True, help="Ruta al modelo YOLO entrenado (1 clase)")
    p.add_argument("--img_dir", type=Path, default=Path("data/raw/unmarked"), help="Imágenes HR originales")
    p.add_argument("--gt_dir", type=Path, default=Path("data/raw/labels"), help="Etiquetas originales (2 clases)")
    p.add_argument("--out_dir", type=Path, default=Path("data/crops"), help="Directorio de salida para los recortes")
    p.add_argument("--img_size", type=int, default=512, help="Resolución de inferencia")
    p.add_argument("--iou_thresh", type=float, default=0.3, help="Umbral de IoU para asignar etiqueta")
    p.add_argument("--margin", type=int, default=10, help="Margen en píxeles al recortar")
    return p.parse_args()

def main():
    args = parse_args()
    
    if not args.model.exists():
        print(f"[ERROR] Modelo no encontrado: {args.model}")
        sys.exit(1)
        
    model = YOLO(str(args.model))
    
    out_normal = args.out_dir / "0_normal"
    out_dicentric = args.out_dir / "1_dicentric"
    out_unlabeled = args.out_dir / "unlabeled" # Para recortes que no casan con ningun GT
    
    out_normal.mkdir(parents=True, exist_ok=True)
    out_dicentric.mkdir(parents=True, exist_ok=True)
    out_unlabeled.mkdir(parents=True, exist_ok=True)
    
    images = list(args.img_dir.glob("*.[jJ][pP][gG]")) + list(args.img_dir.glob("*.png"))
    
    stats = {"0_normal": 0, "1_dicentric": 0, "unlabeled": 0}
    
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        # 1. Preprocesamiento
        img_prep = remove_circular_artifacts(img)
        h_orig, w_orig = img.shape[:2]
        
        # 2. Inferencia YOLO
        results = model.predict(img_prep, imgsz=args.img_size, verbose=False)
        
        # 3. Cargar Ground Truth
        gt_path = args.gt_dir / f"{img_path.stem}.txt"
        gt_boxes = load_gt_labels(gt_path)
        
        predictions = results[0].boxes
        
        for i, box in enumerate(predictions):
            # Coordenadas [x_center, y_center, w, h] normalizadas predichas
            xywhn = box.xywhn[0].cpu().numpy()
            
            # Asignar clase usando IoU
            best_iou = 0
            best_cls = -1
            
            for gt_cls, gt_xywh in gt_boxes:
                iou = calculate_iou(xywhn, gt_xywh)
                if iou > best_iou:
                    best_iou = iou
                    best_cls = gt_cls
                    
            if best_iou >= args.iou_thresh:
                label_name = "0_normal" if best_cls == 0 else "1_dicentric"
            else:
                label_name = "unlabeled"
                
            stats[label_name] += 1
            
            # Recortar la imagen con coordenadas absolutas
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            
            # Añadir margen
            x1 = max(0, x1 - args.margin)
            y1 = max(0, y1 - args.margin)
            x2 = min(w_orig, x2 + args.margin)
            y2 = min(h_orig, y2 + args.margin)
            
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
                
            crop_name = f"{img_path.stem}_crop{i}_{label_name}.png"
            if label_name == "0_normal":
                cv2.imwrite(str(out_normal / crop_name), crop)
            elif label_name == "1_dicentric":
                cv2.imwrite(str(out_dicentric / crop_name), crop)
            else:
                cv2.imwrite(str(out_unlabeled / crop_name), crop)
                
    print("\n[INFO] Extracción de recortes completada.")
    print(f"       Normales   : {stats['0_normal']}")
    print(f"       Dicéntricos: {stats['1_dicentric']}")
    print(f"       Sin etiq.  : {stats['unlabeled']}")
    print(f"[INFO] Recortes guardados en: {args.out_dir}")

if __name__ == "__main__":
    main()
