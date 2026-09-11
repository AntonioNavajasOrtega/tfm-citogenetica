import os
import shutil
import torch
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import yaml

def get_base_imgsz(folder_path):
    """Obtiene el tamaño máximo de las imágenes en la carpeta, múltiplo de 32."""
    max_dim = 0
    for file in folder_path.iterdir():
        if file.suffix.lower() in [".jpg", ".png", ".jpeg", ".tif"]:
            with Image.open(file) as img:
                w, h = img.size
                max_dim = max(max_dim, w, h)
    
    if max_dim == 0:
        return 1280
    
    return ((max_dim + 31) // 32) * 32

def main():
    root_dir = Path("C:/TFM")
    ckpt_path = root_dir / "best.pt"
    
    if not ckpt_path.exists():
        print(f"ERROR: No se encontró el modelo YOLO en {ckpt_path}")
        return

    model = YOLO(str(ckpt_path))
    sr_base_dir = root_dir / "results/sr"
    labels_dir = root_dir / "data/test/labels"
    
    folders_to_eval = [f for f in sr_base_dir.iterdir() if f.is_dir() and any(f.iterdir())]
    print(f"Encontradas {len(folders_to_eval)} carpetas para evaluar.")

    for folder in folders_to_eval:
        print("\n" + "="*50)
        target_imgsz = get_base_imgsz(folder)
        
        # Estrategia de Dispositivo (GPU vs CPU)
        if target_imgsz > 2048:
            eval_device = "cpu"
            print(f"Evaluando carpeta: {folder.name} | imgsz={target_imgsz} | DEVICE=CPU (Para evitar OOM)")
        else:
            eval_device = "0" # GPU 0
            print(f"Evaluando carpeta: {folder.name} | imgsz={target_imgsz} | DEVICE=GPU")
            
        print("="*50)
        
        temp_dataset_dir = root_dir / "temp_yolo_dataset" / folder.name
        temp_images = temp_dataset_dir / "images"
        temp_labels = temp_dataset_dir / "labels"
        
        temp_images.mkdir(parents=True, exist_ok=True)
        temp_labels.mkdir(parents=True, exist_ok=True)
        
        valid_imgs = 0
        for img in folder.glob("*.*"):
            if img.suffix.lower() in [".jpg", ".png", ".jpeg", ".tif"]:
                lbl_name = img.stem + ".txt"
                lbl_src = labels_dir / lbl_name
                if lbl_src.exists():
                    shutil.copy2(img, temp_images / img.name)
                    shutil.copy2(lbl_src, temp_labels / lbl_name)
                    valid_imgs += 1
        
        if valid_imgs == 0:
            print(f"Saltando {folder.name}: no hay imágenes con etiquetas de test.")
            shutil.rmtree(temp_dataset_dir, ignore_errors=True)
            continue

        yaml_path = root_dir / f"temp_{folder.name}.yaml"
        dataset_config = {
            "path": str(temp_dataset_dir),
            "train": "images",
            "val": "images",
            "names": {0: "chromosome", 1: "dicentric"}
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(dataset_config, f)
            
        try:
            results = model.val(
                data=str(yaml_path),
                imgsz=target_imgsz,
                batch=1,
                device=eval_device,
                half=(eval_device != "cpu"), # FP16 crashea en CPU con artefactos RGB
                workers=0, # Desactivar multiprocessing para evitar corrupción de memoria compartida en Windows
                name=f"eval_{folder.name}_{target_imgsz}",
                project="results/yolo_sr_eval"
            )
            print(f"Resultados para {folder.name} (imgsz={target_imgsz}, {eval_device}): mAP50 = {results.box.map50:.4f}")
        except Exception as e:
            print(f"Error evaluando {folder.name}: {e}")
            
        yaml_path.unlink(missing_ok=True)
        shutil.rmtree(temp_dataset_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
