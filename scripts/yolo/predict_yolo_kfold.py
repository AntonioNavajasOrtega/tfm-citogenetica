"""
predict_yolo_kfold.py — Infiere y recorta sobre los validación splits de cada fold.

Requiere haber ejecutado train_yolo_kfold.py primero.
Extrae los recortes y los asigna a 0_normal, 1_dicentric en data/crops_hr/.

Uso:
    python scripts/yolo/predict_yolo_kfold.py
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("Instala ultralytics: pip install ultralytics")
    sys.exit(1)


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    return logging.getLogger(__name__)


def calculate_iou(box1, box2):
    b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1: return 0.0
        
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    return inter_area / (b1_area + b2_area - inter_area)


def load_gt_labels(label_path: Path):
    if not label_path.exists(): return []
    gt_boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            gt_boxes.append((int(parts[0]), [float(p) for p in parts[1:]]))
    return gt_boxes


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folds_json", type=Path, default=Path("results/folds_info.json"))
    p.add_argument("--imgs_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--labels_2class_dir", type=Path, default=Path("data/raw/labels"))
    p.add_argument("--crops_out_dir", type=Path, default=Path("data/crops_hr"))
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--iou_thresh", type=float, default=0.3)
    p.add_argument("--margin", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    if not args.folds_json.exists():
        logger.error(f"Falta {args.folds_json}")
        sys.exit(1)

    folds_info = json.loads(args.folds_json.read_text())

    out_normal = args.crops_out_dir / "0_normal"
    out_dicentric = args.crops_out_dir / "1_dicentric"
    out_unlabeled = args.crops_out_dir / "unlabeled"
    
    if args.crops_out_dir.exists():
        shutil.rmtree(args.crops_out_dir)
    out_normal.mkdir(parents=True)
    out_dicentric.mkdir(parents=True)
    out_unlabeled.mkdir(parents=True)

    crop_stats = {"0_normal": 0, "1_dicentric": 0, "unlabeled": 0}

    for fold_name, f_data in folds_info.items():
        fold_idx = fold_name.replace("fold_", "")
        best_pt = Path("models/yolo_kfold") / fold_name / "train/weights/best.pt"
        
        if not best_pt.exists():
            logger.error(f"Modelo para {fold_name} no encontrado en {best_pt}")
            continue

        logger.info("=" * 60)
        logger.info(f"GENERANDO RECORTES - FOLD {fold_idx}")
        logger.info("=" * 60)
        
        val_model = YOLO(str(best_pt))
        
        for img_name in f_data["val"]:
            img_path = args.imgs_dir / img_name
            if not img_path.exists(): continue
            
            img = cv2.imread(str(img_path))
            h_orig, w_orig = img.shape[:2]
            
            results = val_model.predict(img, imgsz=args.img_size, verbose=False)
            
            gt_file = img_name.replace(Path(img_name).suffix, ".txt")
            gt_boxes = load_gt_labels(args.labels_2class_dir / gt_file)
            
            predictions = results[0].boxes
            for i, box in enumerate(predictions):
                xywhn = box.xywhn[0].cpu().numpy()
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
                    
                crop_stats[label_name] += 1
                
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                x1 = max(0, x1 - args.margin)
                y1 = max(0, y1 - args.margin)
                x2 = min(w_orig, x2 + args.margin)
                y2 = min(h_orig, y2 + args.margin)
                
                crop = img[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                crop_name = f"{Path(img_name).stem}_fold{fold_idx}_crop{i}_{label_name}.png"
                if label_name == "0_normal":
                    cv2.imwrite(str(out_normal / crop_name), crop)
                elif label_name == "1_dicentric":
                    cv2.imwrite(str(out_dicentric / crop_name), crop)
                else:
                    cv2.imwrite(str(out_unlabeled / crop_name), crop)

    logger.info("=" * 60)
    logger.info("EXTRACCIÓN K-FOLD COMPLETADA")
    logger.info(f"  Normales:    {crop_stats['0_normal']}")
    logger.info(f"  Dicéntricos: {crop_stats['1_dicentric']}")
    logger.info(f"  Sin etiquetar: {crop_stats['unlabeled']}")

if __name__ == "__main__":
    main()
