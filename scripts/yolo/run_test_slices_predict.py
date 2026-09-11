import shutil
from pathlib import Path
from ultralytics import YOLO

model_path = Path("C:/TFM/models/yolo_single/yolo11s/train/weights/best.pt")
lbl_dir = Path("C:/TFM/data/test/test_sr_sliced_1280/labels")
out_dir = Path("C:/TFM/results/predictions_test_sr_sliced")

if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

test_imgs = [
    Path("C:/TFM/data/test/test_sr_sliced_1280/images/2Gy-138_slice_1024_1024_2304_2304.png"),
    Path("C:/TFM/data/test/test_sr_sliced_1280/images/2Gy-016_slice_2048_1024_3328_2304.png"),
    Path("C:/TFM/data/test/test_sr_sliced_1280/images/2Gy-065_slice_2048_1024_3328_2304.png"),
    Path("C:/TFM/data/test/test_sr_sliced_1280/images/2Gy-131_slice_1024_1024_2304_2304.png"),
    Path("C:/TFM/data/test/test_sr_sliced_1280/images/2Gy-106_slice_1024_1024_2304_2304.png")
]

model = YOLO(str(model_path))

print("="*80)
print("INFERENCIA YOLO11S SOBRE IMÁGENES TEST SR SLICED (1280x1280)")
print("Modelo:", model_path)
print("Clases del modelo:", model.names)
print("="*80)

for img_p in test_imgs:
    # Ground truth info
    lbl_p = lbl_dir / (img_p.stem + ".txt")
    gt_chr, gt_dic = 0, 0
    if lbl_p.exists():
        for line in lbl_p.read_text().strip().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            c = int(parts[0])
            if c == 0:
                gt_chr += 1
            elif c == 1:
                gt_dic += 1

    res = model.predict(
        source=str(img_p),
        imgsz=1280,
        conf=0.25,
        save=True,
        save_txt=True,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        verbose=False
    )[0]

    boxes = res.boxes
    pred_chr, pred_dic = 0, 0
    dic_confs = []

    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            c = int(b.cls[0])
            conf_val = float(b.conf[0])
            if c == 0:
                pred_chr += 1
            else:
                pred_dic += 1
                dic_confs.append(round(conf_val, 2))

    total_gt = gt_chr + gt_dic
    total_pred = len(boxes) if boxes is not None else 0
    print(f"{img_p.name[:42]:42s} | GT: {total_gt:2d} (chr:{gt_chr:2d}, dic:{gt_dic:1d}) | PRED: {total_pred:2d} (chr:{pred_chr:2d}, dic:{pred_dic:1d}) | Dic confs: {dic_confs}")

print("="*80)
print(f"Predicciones e imágenes guardadas en: {out_dir}")
