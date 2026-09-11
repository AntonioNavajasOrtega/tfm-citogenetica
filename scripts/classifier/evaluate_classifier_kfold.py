"""
evaluate_classifier_kfold.py — Evalúa los modelos entrenados en sus respectivos folds de val.

Lee results/folds_info.json, carga el modelo correspondiente a cada fold,
evalúa sobre los recortes de validación, y consolida las métricas.

Uso:
    python scripts/classifier/evaluate_classifier_kfold.py --dataset data/crops_hr
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

try:
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
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


def get_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def initialize_model(num_classes):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--folds_json", type=Path, default=Path("results/folds_info.json"))
    p.add_argument("--models_dir", type=Path, default=Path("models/classifier"))
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    exp_name = f"cnn_{args.dataset.name}"
    log_file = args.models_dir / f"{exp_name}_eval.log"
    logger = setup_logging(log_file)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if not args.dataset.exists() or not args.folds_json.exists():
        logger.error("Dataset o folds_json no encontrado.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"EVALUACIÓN CLASIFICADOR: {args.dataset.name}")
    logger.info("=" * 60)

    folds_info = json.loads(args.folds_json.read_text())
    dataset_val = datasets.ImageFolder(args.dataset, transform=get_transforms(args.img_size))
    class_names = dataset_val.classes
    samples_paths = [Path(s[0]).name for s in dataset_val.samples]
    
    fold_results = []
    cm_global = np.zeros((len(class_names), len(class_names)), dtype=int)

    for fold_name, f_data in folds_info.items():
        fold_idx = fold_name.replace("fold_", "")
        model_path = args.models_dir / f"{exp_name}_fold{fold_idx}.pth"
        
        if not model_path.exists():
            logger.error(f"Modelo {model_path} no encontrado.")
            continue

        val_orig_imgs = set([Path(p).stem for p in f_data["val"]])
        val_idx = [i for i, crop in enumerate(samples_paths) if any(crop.startswith(o) for o in val_orig_imgs)]
        
        if not val_idx: continue
            
        val_sub = Subset(dataset_val, val_idx)
        dataloader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False)
        
        model = initialize_model(len(class_names)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        
        cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
        cm_global += cm
        
        fold_results.append({"fold": fold_idx, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
        logger.info(f"Fold {fold_idx} | Acc: {acc:.4f} | F1: {f1:.4f}")

    if fold_results:
        csv_path = args.models_dir / f"{exp_name}_metrics.csv"
        keys = ["accuracy", "precision", "recall", "f1"]
        means = {k: np.mean([r[k] for r in fold_results]) for k in keys}
        stds = {k: np.std([r[k] for r in fold_results]) for k in keys}
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['fold'] + keys)
            writer.writeheader()
            for r in fold_results:
                writer.writerow(r)
            writer.writerow({'fold': 'MEDIA', **means})
            writer.writerow({'fold': 'STD', **stds})
            
        logger.info("=" * 60)
        logger.info(" RESULTADOS FINALES GLOBALES ")
        logger.info("=" * 60)
        logger.info(f"MEDIA F1: {means['f1']:.4f} ± {stds['f1']:.4f}")
        logger.info(f"MEDIA Acc: {means['accuracy']:.4f} ± {stds['accuracy']:.4f}")
        logger.info("\nMATRIZ DE CONFUSIÓN ACUMULADA:")
        logger.info(f"\n{cm_global}")
        logger.info(f"\nMétricas guardadas en {csv_path}")

if __name__ == "__main__":
    main()
