"""
train_classifier_kfold.py — Entrena CNN para clasificar recortes en K folds.

Utiliza los folds estáticos definidos en results/folds_info.json.
Guarda los modelos entrenados en models/classifier/.

Uso:
    python scripts/classifier/train_classifier_kfold.py --dataset data/crops_hr --pretrained
"""

import argparse
import copy
import json
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
    from sklearn.metrics import accuracy_score
except ImportError:
    print("Instala scikit-learn: pip install scikit-learn")
    sys.exit(1)


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
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
    return {
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


def initialize_model(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, fold, logger):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
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

            epoch_loss = running_loss / max(1, len(dataloaders[phase].dataset))
            epoch_acc = accuracy_score(all_labels, all_preds)
            
            if phase == 'val':
                if epoch % 5 == 0 or epoch == num_epochs - 1:
                    logger.info(f'Fold {fold} - Epoch {epoch}/{num_epochs - 1} | Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    logger.info(f'Fold {fold} - Entrenamiento completado en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Fold {fold} - Mejor Val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True, help="Ruta a data/crops_hr o data/crops_sr")
    p.add_argument("--folds_json", type=Path, default=Path("results/folds_info.json"))
    p.add_argument("--output_dir", type=Path, default=Path("models/classifier"))
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp_name = f"cnn_{args.dataset.name}"
    
    log_file = args.output_dir / f"{exp_name}_train.log"
    logger = setup_logging(log_file)
    set_seed(args.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if not args.dataset.exists():
        logger.error(f"Error: No existe el dataset {args.dataset}")
        sys.exit(1)
    if not args.folds_json.exists():
        logger.error(f"Error: No existe json de folds {args.folds_json}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"ENTRENAMIENTO CLASIFICADOR: {args.dataset.name}")
    logger.info("=" * 60)

    with open(args.folds_json, "r") as f:
        folds_info = json.load(f)

    transforms_dict = get_transforms(args.img_size)
    dataset_train = datasets.ImageFolder(args.dataset, transform=transforms_dict['train'])
    dataset_val = datasets.ImageFolder(args.dataset, transform=transforms_dict['val'])
    
    class_names = dataset_train.classes

    samples_paths = [Path(s[0]).name for s in dataset_train.samples]

    for fold_name, f_data in folds_info.items():
        fold_idx = fold_name.replace("fold_", "")
        logger.info(f"\n--- {fold_name.upper()} ---")
        
        val_orig_imgs = set([Path(p).stem for p in f_data["val"]])
        
        train_idx, val_idx = [], []
        
        for i, crop_name in enumerate(samples_paths):
            is_val = any(crop_name.startswith(orig) for orig in val_orig_imgs)
            if is_val: val_idx.append(i)
            else: train_idx.append(i)
                
        if len(train_idx) == 0 or len(val_idx) == 0:
            logger.warning(f"Fold {fold_idx} vacío, ignorando...")
            continue
            
        train_sub = Subset(dataset_train, train_idx)
        val_sub = Subset(dataset_val, val_idx)
        
        dataloaders = {
            'train': DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, num_workers=0),
            'val': DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, num_workers=0)
        }
        
        model = initialize_model(len(class_names), pretrained=args.pretrained).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        best_model = train_model(model, dataloaders, criterion, optimizer, args.epochs, device, fold_idx, logger)
        
        model_path = args.output_dir / f"{exp_name}_fold{fold_idx}.pth"
        torch.save(best_model.state_dict(), model_path)
        logger.info(f"Modelo guardado en {model_path}")

    logger.info("\n[INFO] Entrenamiento K-Fold de Clasificador completado.")

if __name__ == "__main__":
    main()
