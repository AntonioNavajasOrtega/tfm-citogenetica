"""
train_resshift_kfold.py — Hace finetuning de ResShift en 5 folds estáticos.

Lee results/folds_info.json.
Para cada fold i:
  1. Crea temporalmente una carpeta con los recortes LR y HR de los otros 4 folds.
  2. Llama a train_resshift.py para hacer finetuning.
  3. El modelo se guarda en models/resshift_finetuned/

Uso:
    python scripts/diffusion/train_resshift_kfold.py --iterations 2000
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
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--scale", type=int, default=2)
    return p.parse_args()


def prepare_fold_data(fold_name, folds_info, crops_hr_dir, crops_lr_dir, temp_dir):
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        
    train_hr = temp_dir / "train" / "hr"
    train_lr = temp_dir / "train" / "lr"
    
    train_hr.mkdir(parents=True)
    train_lr.mkdir(parents=True)
    
    val_orig_imgs = set(folds_info[fold_name]["val"])
    
    for subdir in crops_hr_dir.iterdir():
        if not subdir.is_dir(): continue
        class_name = subdir.name
        
        for crop_path in subdir.glob("*.png"):
            is_val = False
            for orig_val in val_orig_imgs:
                orig_stem = Path(orig_val).stem
                if crop_path.name.startswith(orig_stem):
                    is_val = True
                    break
                    
            if not is_val:
                shutil.copy2(crop_path, train_hr / crop_path.name)
                lr_crop_path = crops_lr_dir / class_name / crop_path.name
                if lr_crop_path.exists():
                    shutil.copy2(lr_crop_path, train_lr / crop_path.name)
                    
    return train_hr, train_lr


def main():
    args = parse_args()
    
    if not args.folds_json.exists():
        print(f"[ERROR] No se encuentra {args.folds_json}")
        sys.exit(1)
        
    with open(args.folds_json, "r") as f:
        folds_info = json.load(f)
        
    temp_dir = Path("data/temp_resshift_train")
    
    for fold_name in folds_info.keys():
        print("=" * 60)
        print(f"RES-SHIFT ENTRENAMIENTO: Procesando {fold_name}")
        print("=" * 60)
        
        train_hr, train_lr = prepare_fold_data(
            fold_name, folds_info, args.crops_hr_dir, args.crops_lr_dir, temp_dir
        )
        
        exp_name = f"resshift_{fold_name}"
        train_cmd = [
            sys.executable, "scripts/diffusion/train_resshift.py",
            "--hr_dir", str(train_hr),
            "--lr_dir", str(train_lr),
            "--scale", str(args.scale),
            "--iterations", str(args.iterations),
            "--exp_name", exp_name
        ]
        
        print(f"Ejecutando Finetuning: {' '.join(train_cmd)}")
        subprocess.run(train_cmd, check=True)
                
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        
    print("\n[INFO] Entrenamiento K-Fold de ResShift completado.")


if __name__ == "__main__":
    main()
