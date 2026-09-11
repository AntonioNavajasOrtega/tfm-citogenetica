"""
eval_classifier.py — Evalúa un modelo CNN entrenado sobre una carpeta de recortes.

Se utiliza para comparar el rendimiento del clasificador en recortes originales
versus recortes superresueltos (SR).

Uso:
    python scripts/classifier/eval_classifier.py \
        --data_dir data/crops_sr \
        --model_path models/classifier/cnn_classifier_fold1.pth \
        --pretrained
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    print("Instala scikit-learn: pip install scikit-learn")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Evalúa CNN en recortes de cromosomas")
    p.add_argument("--data_dir", type=Path, required=True, help="Directorio con recortes a evaluar")
    p.add_argument("--model_path", type=Path, required=True, help="Ruta al modelo entrenado (.pth)")
    p.add_argument("--arch", type=str, default="auto", choices=["auto", "densenet121", "resnet18"],
                   help="Arquitectura de la red (def: auto-detectar entre densenet121 y resnet18)")
    p.add_argument("--img_size", type=int, default=128, help="Resolución esperada por la CNN")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--pretrained", action="store_true", help="Si el modelo base era preentrenado")
    p.add_argument("--save_visual", type=Path, default=None,
                   help="Directorio donde guardar imágenes visuales con predicción y Grad-CAM")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.data_dir.exists():
        print(f"[ERROR] Directorio de datos no existe: {args.data_dir}")
        sys.exit(1)

    if not args.model_path.exists():
        print(f"[ERROR] Modelo no existe: {args.model_path}")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Transformaciones de inferencia
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    class_names = dataset.classes
    num_classes = len(class_names)
    
    if num_classes != 2:
        print(f"[WARNING] Se encontraron {num_classes} clases ({class_names}). Se esperan 2 (normal, dicentric).")

    # Cargar modelo con soporte para DenseNet121 y ResNet18
    state_dict = torch.load(args.model_path, map_location=device)
    arch = args.arch
    if arch == "auto":
        keys_str = " ".join(state_dict.keys())
        if "features.denseblock" in keys_str or "features.norm5" in keys_str:
            arch = "densenet121"
        else:
            arch = "resnet18"

    if arch == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if args.pretrained else None
        model = models.densenet121(weights=weights)
        in_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(in_ftrs, num_classes)
    else:
        weights = models.ResNet18_Weights.DEFAULT if args.pretrained else None
        model = models.resnet18(weights=weights)
        in_ftrs = model.fc.in_features
        model.fc = nn.Linear(in_ftrs, num_classes)
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print(f"\nEvaluando modelo {args.model_path.name} sobre {args.data_dir.name}...")
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n" + "="*50)
    print(" REPORTE DE CLASIFICACIÓN ")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    print(" MATRIZ DE CONFUSIÓN ")
    print("="*50)
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print("="*50 + "\n")

    if args.save_visual:
        sys.path.append(str(Path(__file__).resolve().parents[2]))
        from scripts.classifier.predict_visual import run_inference_and_gradcam, create_visual_card, create_mosaic_grid
        from PIL import Image
        import cv2

        args.save_visual.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Generando visualizaciones y Grad-CAM en: {args.save_visual}...")

        target_layer = model.features.norm5 if arch == "densenet121" else model.layer4[-1]
        cards = []
        samples_to_visualize = dataset.samples[:40]  # hasta 40 muestras representativas

        for img_path_str, label in samples_to_visualize:
            p = Path(img_path_str)
            try:
                img_pil = Image.open(p).convert("RGB")
            except Exception:
                continue

            pred_name, conf, _, cam_overlay = run_inference_and_gradcam(
                model, target_layer, img_pil, args.img_size, device, use_gradcam=True
            )
            card = create_visual_card(img_pil, pred_name, conf, cam_overlay, args.img_size)
            cards.append(card)

            out_name = f"eval_{pred_name}_{int(conf*100)}_{p.stem}.png"
            cv2.imwrite(str(args.save_visual / out_name), card)

        if cards:
            grid = create_mosaic_grid(cards, cols=4)
            if grid is not None:
                cv2.imwrite(str(args.save_visual / "summary_grid.png"), grid)
                print(f"[INFO] Mosaico de evaluación guardado en: {args.save_visual / 'summary_grid.png'}")

        print(f"[INFO] Visualizaciones completadas.")


if __name__ == "__main__":
    main()
