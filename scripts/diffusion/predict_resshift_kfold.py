"""
predict_resshift_kfold.py — Aplica SR (run_resshift.py) a los recortes del fold de validación
usando el modelo recién entrenado para ese fold.
Guarda el resultado en data/crops_sr/.

Requiere haber ejecutado train_resshift_kfold.py primero.

Uso:
    python scripts/diffusion/predict_resshift_kfold.py
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--folds_json", type=Path, default=Path("results/folds_info.json"))
    p.add_argument("--crops_hr_dir", type=Path, default=Path("data/crops_hr"))
    p.add_argument("--crops_lr_dir", type=Path, default=Path("data/crops_lr"))
    p.add_argument("--crops_sr_dir", type=Path, default=Path("data/crops_sr"))
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--version", type=str, default="v3")
    return p.parse_args()


def prepare_val_data(fold_name, folds_info, crops_hr_dir, crops_lr_dir, temp_dir):
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        
    val_lr = temp_dir / "val" / "lr"
    val_lr.mkdir(parents=True)
    
    val_orig_imgs = set(folds_info[fold_name]["val"])
    
    for subdir in crops_hr_dir.iterdir():
        if not subdir.is_dir(): continue
        class_name = subdir.name
        
        val_lr_cls = val_lr / class_name
        val_lr_cls.mkdir(exist_ok=True)
        
        for crop_path in subdir.glob("*.png"):
            is_val = False
            for orig_val in val_orig_imgs:
                orig_stem = Path(orig_val).stem
                if crop_path.name.startswith(orig_stem):
                    is_val = True
                    break
                    
            lr_crop_path = crops_lr_dir / class_name / crop_path.name
            
            if is_val:
                if lr_crop_path.exists():
                    shutil.copy2(lr_crop_path, val_lr_cls / crop_path.name)
                    
    return val_lr


def main():
    args = parse_args()
    
    if not args.folds_json.exists():
        print(f"[ERROR] No se encuentra {args.folds_json}")
        sys.exit(1)
        
    with open(args.folds_json, "r") as f:
        folds_info = json.load(f)
        
    if args.crops_sr_dir.exists():
        shutil.rmtree(args.crops_sr_dir)
    args.crops_sr_dir.mkdir(parents=True)
    
    temp_dir = Path("data/temp_resshift_infer")
    
    for fold_name in folds_info.keys():
        print("=" * 60)
        print(f"RES-SHIFT INFERENCIA: Procesando {fold_name}")
        print("=" * 60)
        
        val_lr = prepare_val_data(
            fold_name, folds_info, args.crops_hr_dir, args.crops_lr_dir, temp_dir
        )
        
        exp_name = f"resshift_{fold_name}"
        ckpt_path = Path("models/resshift_finetuned") / exp_name / "ckpts" / "last.pth"
        if not ckpt_path.exists():
            print(f"[ERROR] No se encontró el checkpoint entrenado en {ckpt_path}")
            continue
            
        for class_dir in val_lr.iterdir():
            if not class_dir.is_dir(): continue
            class_name = class_dir.name
            out_class_dir = args.crops_sr_dir / class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)
            
            infer_cmd = [
                sys.executable, "scripts/diffusion/run_resshift.py",
                "--input_dir", str(class_dir),
                "--output_dir", str(out_class_dir),
                "--scale", str(args.scale),
                "--version", args.version,
                "--ckpt_path", str(ckpt_path),
                "--exp_name", f"temp_{fold_name}_{class_name}"
            ]
            
            subprocess.run(infer_cmd, check=True)
            
            temp_out = out_class_dir / f"temp_{fold_name}_{class_name}"
            if temp_out.exists():
                for sr_img in temp_out.glob("*.*"):
                    shutil.move(str(sr_img), str(out_class_dir / sr_img.name))
                shutil.rmtree(temp_out)
                
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        
    print("\n[INFO] Inferencia K-Fold de ResShift completada.")
    print(f"[INFO] Recortes SR guardados en {args.crops_sr_dir}")


if __name__ == "__main__":
    main()
