"""
eval_sr_metrics.py — Calcula métricas de calidad de imagen (PSNR, SSIM, LPIPS)
comparando los recortes HR originales con los generados por Superresolución (SR).

Uso:
    python scripts/diffusion/eval_sr_metrics.py --hr_dir data/crops_hr --sr_dir data/crops_sr
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    import lpips
except ImportError:
    print("Instala dependencias: pip install scikit-image lpips")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hr_dir", type=Path, default=Path("data/crops_hr"))
    p.add_argument("--sr_dir", type=Path, default=Path("data/crops_sr"))
    return p.parse_args()


def load_img_for_lpips(img):
    # LPIPS espera tensor RGB en rango [-1, 1] con formato NCHW
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    img_tensor = img_tensor * 2 - 1
    return img_tensor.permute(2, 0, 1).unsqueeze(0)


def main():
    args = parse_args()
    
    if not args.hr_dir.exists() or not args.sr_dir.exists():
        print("[ERROR] Asegúrate de que las carpetas hr_dir y sr_dir existan.")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    metrics = {"psnr": [], "ssim": [], "lpips": []}
    
    print(f"Evaluando {args.sr_dir.name} frente a {args.hr_dir.name}...\n")
    
    subdirs = [d for d in args.sr_dir.iterdir() if d.is_dir()]
    
    for subdir in subdirs:
        class_name = subdir.name
        hr_subdir = args.hr_dir / class_name
        
        if not hr_subdir.exists():
            continue
            
        for sr_img_path in subdir.glob("*.png"):
            hr_img_path = hr_subdir / sr_img_path.name
            if not hr_img_path.exists():
                continue
                
            img_sr = cv2.imread(str(sr_img_path))
            img_hr = cv2.imread(str(hr_img_path))
            
            if img_sr is None or img_hr is None:
                continue
                
            # Igualar resoluciones para la métrica si ResShift hizo upscale > 1
            # (Lo ideal es comparar a la resolución del HR)
            h, w = img_hr.shape[:2]
            if img_sr.shape[:2] != (h, w):
                img_sr = cv2.resize(img_sr, (w, h), interpolation=cv2.INTER_CUBIC)
            
            # PSNR & SSIM
            p_val = psnr(img_hr, img_sr)
            s_val = ssim(img_hr, img_sr, channel_axis=2)
            
            # LPIPS
            t_hr = load_img_for_lpips(img_hr).to(device)
            t_sr = load_img_for_lpips(img_sr).to(device)
            with torch.no_grad():
                l_val = loss_fn_vgg(t_hr, t_sr).item()
                
            metrics["psnr"].append(p_val)
            metrics["ssim"].append(s_val)
            metrics["lpips"].append(l_val)
            
    if not metrics["psnr"]:
        print("[WARNING] No se encontraron imágenes emparejadas para evaluar.")
        sys.exit(0)
        
    mean_psnr = np.mean(metrics["psnr"])
    mean_ssim = np.mean(metrics["ssim"])
    mean_lpips = np.mean(metrics["lpips"])
    
    print("=" * 40)
    print(" RESULTADOS GLOBALES DE CALIDAD (SR)")
    print("=" * 40)
    print(f" Imágenes evaluadas : {len(metrics['psnr'])}")
    print(f" PSNR  (↑ mejor)    : {mean_psnr:.4f} dB")
    print(f" SSIM  (↑ mejor)    : {mean_ssim:.4f}")
    print(f" LPIPS (↓ mejor)    : {mean_lpips:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
