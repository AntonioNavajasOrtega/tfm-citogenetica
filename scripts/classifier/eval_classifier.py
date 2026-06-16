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
    p.add_argument("--img_size", type=int, default=128, help="Resolución esperada por la CNN")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--pretrained", action="store_true", help="Si el modelo base era preentrenado")
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

    # Cargar modelo
    weights = models.ResNet18_Weights.DEFAULT if args.pretrained else None
    model = models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
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

if __name__ == "__main__":
    main()
