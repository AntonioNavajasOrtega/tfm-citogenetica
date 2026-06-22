"""
predict_yolo_kfold.py — Infiere con el modelo de cada fold sobre su conjunto de validación.

Requiere haber ejecutado generate_folds.py y train_yolo_kfold.py primero.
Lee las imágenes de validación directamente desde las carpetas de cada fold
(fold_N/val/images/) en lugar de un splits.json.

Extrae recortes y los clasifica en 0_normal / 1_dicentric / unlabeled
usando las etiquetas GT de 2 clases mediante IoU.

Uso:
    python scripts/yolo/predict_yolo_kfold.py --model yolo11s.pt
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import cv2

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

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area)


def load_gt_labels(label_path: Path):
    if not label_path.exists():
        return []
    gt_boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            gt_boxes.append((int(parts[0]), [float(p) for p in parts[1:]]))
    return gt_boxes


def parse_args():
    p = argparse.ArgumentParser(
        description="Inferencia K-Fold YOLO leyendo folds pre-generados."
    )
    p.add_argument("--folds_dir",       type=Path, default=Path("data/yolo_kfold"),
                   help="Directorio raíz con fold_1/, fold_2/, … (salida de generate_folds.py).")
    p.add_argument("--model",           type=str,  default="yolo11n.pt",
                   help="Modelo usado en el entrenamiento (ej. yolo11s.pt). "
                        "Define dónde se buscan los pesos: models/yolo_kfold/{model_stem}/fold_N/")
    p.add_argument("--labels_2class_dir", type=Path, default=Path("data/raw/labels"),
                   help="Directorio con etiquetas GT de 2 clases (para asignar normal/dicéntrico).")
    p.add_argument("--crops_out_dir",   type=Path, default=Path("data/crops_hr"),
                   help="Directorio de salida de los recortes.")
    p.add_argument("--img_size",        type=int,  default=512)
    p.add_argument("--iou_thresh",      type=float, default=0.3,
                   help="IoU mínimo para asignar etiqueta GT a un crop.")
    p.add_argument("--margin",          type=int,  default=10,
                   help="Píxeles de margen alrededor del bounding box.")
    p.add_argument("--folds",           type=int,  nargs="+", default=None,
                   help="Procesar solo estos índices de fold (ej. --folds 1 3).")
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()
    model_stem = Path(args.model).stem  # e.g. "yolo11s"

    # ── Validar entrada ────────────────────────────────────────────────────────
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

    # ── Preparar directorio de salida ──────────────────────────────────────────
    out_normal    = args.crops_out_dir / "0_normal"
    out_dicentric = args.crops_out_dir / "1_dicentric"
    out_unlabeled = args.crops_out_dir / "unlabeled"

    if args.crops_out_dir.exists():
        shutil.rmtree(args.crops_out_dir)
    out_normal.mkdir(parents=True)
    out_dicentric.mkdir(parents=True)
    out_unlabeled.mkdir(parents=True)

    crop_stats = {"0_normal": 0, "1_dicentric": 0, "unlabeled": 0}

    # ── Procesar cada fold ─────────────────────────────────────────────────────
    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        fold_idx  = fold_name.replace("fold_", "")

        fold_project_dir = Path("models/yolo_kfold") / model_stem / fold_name
        # Buscar todos los best.pt en subcarpetas tipo train*/weights/best.pt
        best_pt_candidates = list(fold_project_dir.glob("train*/weights/best.pt"))
        
        if not best_pt_candidates:
            logger.error(f"Modelo para {fold_name} no encontrado en {fold_project_dir}\\train*\\weights\\best.pt")
            continue
            
        # Tomar el modificado más recientemente por si hay train, train2, train3...
        best_pt = max(best_pt_candidates, key=lambda p: p.stat().st_mtime)
        logger.info(f"Usando pesos: {best_pt}")

        val_images_dir = fold_dir / "val" / "images"
        if not val_images_dir.exists():
            logger.error(
                f"No se encontró el directorio de validación: {val_images_dir}\n"
                "Comprueba que generate_folds.py se ejecutó correctamente."
            )
            continue

        val_imgs = sorted(
            p for p in val_images_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        )
        if not val_imgs:
            logger.warning(f"[{fold_name}] No hay imágenes de validación en {val_images_dir}")
            continue

        logger.info("=" * 60)
        logger.info(f"GENERANDO RECORTES — FOLD {fold_idx}  ({len(val_imgs)} imágenes)")
        logger.info("=" * 60)

        val_model = YOLO(str(best_pt))

        for img_path in val_imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"No se pudo leer: {img_path}")
                continue

            h_orig, w_orig = img.shape[:2]
            results = val_model.predict(img, imgsz=args.img_size, verbose=False)

            gt_file = img_path.stem + ".txt"
            gt_boxes = load_gt_labels(args.labels_2class_dir / gt_file)

            predictions = results[0].boxes
            for i, box in enumerate(predictions):
                xywhn    = box.xywhn[0].cpu().numpy()
                best_iou = 0.0
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
                if crop.size == 0:
                    continue

                crop_name = f"{img_path.stem}_fold{fold_idx}_crop{i}_{label_name}.png"
                dest_dir = {
                    "0_normal":    out_normal,
                    "1_dicentric": out_dicentric,
                    "unlabeled":   out_unlabeled,
                }[label_name]
                cv2.imwrite(str(dest_dir / crop_name), crop)

    logger.info("=" * 60)
    logger.info("EXTRACCIÓN K-FOLD COMPLETADA")
    logger.info(f"  Normales:      {crop_stats['0_normal']}")
    logger.info(f"  Dicéntricos:   {crop_stats['1_dicentric']}")
    logger.info(f"  Sin etiquetar: {crop_stats['unlabeled']}")


if __name__ == "__main__":
    main()
