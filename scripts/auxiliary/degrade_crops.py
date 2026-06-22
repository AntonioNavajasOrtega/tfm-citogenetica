"""
degrade_crops.py — Simula imágenes de baja resolución (LR) a partir de los recortes HR.

Mantiene la misma estructura de carpetas que data/crops_hr.
Aplica desenfoque gaussiano, ruido y reducción de escala.

Uso:
    python scripts/auxiliary/degrade_crops.py --scale 2
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hr_dir", type=Path, default=Path("data/crops_hr"))
    p.add_argument("--lr_dir", type=Path, default=Path("data/crops_lr"))
    p.add_argument("--scale", type=int, default=2)
    return p.parse_args()


def degrade_image(img, scale):
    h, w = img.shape[:2]
    blur_k = 3 + (scale - 2)
    if blur_k % 2 == 0:
        blur_k += 1
        
    img_blur = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
    noise = np.random.normal(0, 5.0, img_blur.shape).astype(np.float32)
    img_noisy = np.clip(img_blur.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    lr = cv2.resize(img_noisy, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
    return lr


def main():
    args = parse_args()
    
    if not args.hr_dir.exists():
        print(f"[ERROR] El directorio HR {args.hr_dir} no existe.")
        sys.exit(1)
        
    args.lr_dir.mkdir(parents=True, exist_ok=True)
    
    subdirs = [d for d in args.hr_dir.iterdir() if d.is_dir()]
    processed = 0
    
    for subdir in subdirs:
        out_subdir = args.lr_dir / subdir.name
        out_subdir.mkdir(exist_ok=True)
        
        imgs = list(subdir.glob("*.png")) + list(subdir.glob("*.jpg"))
        for img_path in imgs:
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            lr_img = degrade_image(img, args.scale)
            cv2.imwrite(str(out_subdir / img_path.name), lr_img)
            processed += 1
            
    print(f"[INFO] Degradación (x{args.scale}) completada.")
    print(f"[INFO] {processed} recortes guardados en {args.lr_dir}")


if __name__ == "__main__":
    main()
