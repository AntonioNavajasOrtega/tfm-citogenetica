"""
train_classifier.py — Entrena una CNN para clasificar recortes de cromosomas (Normal vs Dicéntrico).

Utiliza K-Fold Cross-Validation.
Permite entrenar desde cero o hacer fine-tuning desde pesos preentrenados en ImageNet.

Uso:
    # Fine-tuning con K-Fold
    python scripts/classifier/train_classifier.py --data_dir data/crops --pretrained
    
    # Desde cero con K-Fold
    python scripts/classifier/train_classifier.py --data_dir data/crops
"""

import argparse
import copy
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

try:
    from sklearn.model_selection import KFold
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
except ImportError:
    print("Instala scikit-learn: pip install scikit-learn")
    sys.exit(1)


def setup_logging(log_path: Path, level: str = "INFO") -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms(img_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms


def initialize_model(num_classes, pretrained=True):
    # Usamos ResNet18 por ser ligero y adecuado para recortes
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Cambiar la capa final
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, fold):
    logger = logging.getLogger(__name__)
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_metrics = {}

    for epoch in range(num_epochs):
        # Cada epoch tiene fase de train y val
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)
            
            if phase == 'val':
                prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
                
                # Imprimir menos verbose si hay muchas epochs
                if epoch % 5 == 0 or epoch == num_epochs - 1:
                    logger.info(f'Fold {fold} - Epoch {epoch}/{num_epochs - 1} | Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {f1:.4f}')

                # Guardar el modelo con mejor Acc (o F1)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_metrics = {
                        "accuracy": epoch_acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1
                    }

    time_elapsed = time.time() - since
    logger.info(f'Fold {fold} - Entrenamiento completado en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Fold {fold} - Mejor Val Acc: {best_acc:4f}')

    # Cargar los mejores pesos
    model.load_state_dict(best_model_wts)
    return model, best_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento de clasificador de recortes")
    p.add_argument("--data_dir", type=Path, default=Path("data/crops"), help="Directorio con recortes (0_normal, 1_dicentric)")
    p.add_argument("--output_dir", type=Path, default=Path("models/classifier"), help="Donde guardar modelos y métricas")
    p.add_argument("--img_size", type=int, default=128, help="Tamaño de imagen para la CNN")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--pretrained", action="store_true", help="Usar pesos preentrenados de ImageNet")
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exp_name", default="cnn_classifier")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = args.output_dir / f"{args.exp_name}.log"
    logger = setup_logging(log_file)
    set_seed(args.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.info("=" * 60)
    logger.info(f"ENTRENAMIENTO CLASIFICADOR (Pretrained={args.pretrained})")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info(f"  Device: {device}")
    logger.info("=" * 60)

    if not args.data_dir.exists():
        logger.error(f"Error: No existe el directorio de datos {args.data_dir}")
        sys.exit(1)

    # El dataset base sin transformaciones (las aplicaremos dinámicamente)
    # Sin embargo, ImageFolder espera transforms, aplicamos solo ToTensor aquí y 
    # la normalización se hace en el DataLoader/Transforms específicos luego.
    # Un truco es crear un wrapper o aplicar la transformación 'train'/'val' luego, pero
    # para simplificar, cargamos dos datasets con distintas transformaciones y 
    # hacemos un Subset de ellos usando los mismos índices.
    
    transforms_dict = get_transforms(args.img_size)
    dataset_train = datasets.ImageFolder(args.data_dir, transform=transforms_dict['train'])
    dataset_val = datasets.ImageFolder(args.data_dir, transform=transforms_dict['val'])
    
    class_names = dataset_train.classes
    num_classes = len(class_names)
    logger.info(f"Clases encontradas: {class_names}")

    # Ignorar carpetas que no son "0_normal" o "1_dicentric" si usamos un subset prefiltrado
    # Para simplicidad, asumimos que solo están esas carpetas, o filtramos "unlabeled".
    if "unlabeled" in class_names:
        logger.error("Aviso: El directorio contiene 'unlabeled'. Elimina o mueve esa carpeta antes de entrenar.")
        sys.exit(1)

    indices = np.arange(len(dataset_train))
    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        fold += 1
        logger.info(f"\n--- FOLD {fold} ---")
        
        train_sub = Subset(dataset_train, train_idx)
        val_sub = Subset(dataset_val, val_idx)
        
        dataloaders = {
            'train': DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, num_workers=0),
            'val': DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, num_workers=0)
        }
        
        model = initialize_model(num_classes, pretrained=args.pretrained)
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        best_model, best_metrics = train_model(
            model, dataloaders, criterion, optimizer, args.epochs, device, fold
        )
        
        best_metrics['fold'] = fold
        fold_results.append(best_metrics)
        
        # Guardar el mejor modelo del fold
        model_path = args.output_dir / f"{args.exp_name}_fold{fold}.pth"
        torch.save(best_model.state_dict(), model_path)
        logger.info(f"Modelo guardado en {model_path}")

    # Resumen CSV
    csv_path = args.output_dir / f"{args.exp_name}_metrics.csv"
    keys = ["accuracy", "precision", "recall", "f1"]
    
    means = {k: np.mean([r[k] for r in fold_results]) for k in keys}
    stds = {k: np.std([r[k] for r in fold_results]) for k in keys}
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['fold'] + keys)
        writer.writeheader()
        for r in fold_results:
            writer.writerow({k: r[k] for k in ['fold'] + keys})
        writer.writerow({'fold': 'MEDIA', **means})
        writer.writerow({'fold': 'STD', **stds})
        
    logger.info(f"\nResultados guardados en {csv_path}")
    logger.info(f"MEDIA F1: {means['f1']:.4f} ± {stds['f1']:.4f}")

if __name__ == "__main__":
    main()
