"""
predict_visual.py — Inferencia visual para clasificadores de cromosomas (DenseNet121 / ResNet18).

Permite que el clasificador devuelva imágenes con:
1. Etiqueta visual y porcentaje de confianza (color verde si normal, rojo/naranja si dicéntrico).
2. Mapa de calor Grad-CAM superpuesto (muestra dónde se fijó la red neuronal).
3. Opcionalmente, un mosaico resumen (grid) con múltiples predicciones.

Uso:
    # Para una imagen individual:
    python scripts/classifier/predict_visual.py \
        --model models/classifier_sr/cnn_densenet_folds_sr_fold0.pth \
        --source data/densenet_folds_sr/fold_0/val/dicentric/ejemplo.png

    # Para un directorio de recortes:
    python scripts/classifier/predict_visual.py \
        --model models/classifier_sr/cnn_densenet_folds_sr_fold0.pth \
        --source data/densenet_folds_sr/fold_0/val \
        --max_samples 12 --grid
"""

import argparse
from pathlib import Path
import sys
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


def parse_args():
    p = argparse.ArgumentParser(description="Inferencia visual con DenseNet / ResNet y Grad-CAM")
    p.add_argument("--model", type=Path, required=True, help="Ruta a los pesos del modelo (.pth)")
    p.add_argument("--source", type=Path, required=True, help="Ruta a una imagen o a una carpeta de recortes")
    p.add_argument("--output_dir", type=Path, default=Path("results/classifier_visual"),
                   help="Carpeta donde se guardarán las imágenes generadas")
    p.add_argument("--arch", type=str, default="auto", choices=["auto", "densenet121", "resnet18"],
                   help="Arquitectura de la red (por defecto auto-detecta entre densenet121 y resnet18)")
    p.add_argument("--img_size", type=int, default=128, help="Resolución de entrada a la CNN (def: 128)")
    p.add_argument("--no_gradcam", action="store_true", help="Si se especifica, no genera el mapa de calor Grad-CAM")
    p.add_argument("--grid", action="store_true", help="Generar además un mosaico con todas las predicciones")
    p.add_argument("--max_samples", type=int, default=20,
                   help="Número máximo de imágenes a procesar si el origen es una carpeta (def: 20, usa 0 para todas)")
    return p.parse_args()


def load_classifier(model_path: Path, arch: str, device: torch.device):
    state_dict = torch.load(model_path, map_location=device)

    # Auto-detección de arquitectura
    if arch == "auto":
        keys_str = " ".join(state_dict.keys())
        if "features.denseblock" in keys_str or "features.norm5" in keys_str:
            arch = "densenet121"
        elif "layer4" in keys_str:
            arch = "resnet18"
        else:
            arch = "densenet121"

    if arch == "densenet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 2)
        target_layer = model.features.norm5
    elif arch == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        target_layer = model.layer4[-1]
    else:
        raise ValueError(f"Arquitectura no soportada: {arch}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, arch, target_layer


def run_inference_and_gradcam(model, target_layer, img_pil, img_size, device, use_gradcam=True):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    inp = tf(img_pil).unsqueeze(0).to(device)

    activations = []
    gradients = []

    def f_hook(module, input, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    hook_handle = None
    if use_gradcam:
        hook_handle = target_layer.register_forward_hook(f_hook)

    out = model(inp)
    probs = F.softmax(out, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    conf = float(probs[pred_idx].item())

    classes = ["dicentric", "normal"]
    pred_name = classes[pred_idx]

    cam_overlay = None

    if use_gradcam:
        model.zero_grad()
        out[0, pred_idx].backward()

        if activations and gradients:
            act = activations[0].detach()
            grad = gradients[0].detach()
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * act, dim=1).squeeze().clamp(min=0)
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()
            cam_np = cam.cpu().numpy()

            cam_resized = cv2.resize(cam_np, (img_size, img_size))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

            orig_resized = cv2.resize(np.array(img_pil.convert("RGB")), (img_size, img_size))
            orig_bgr = cv2.cvtColor(orig_resized, cv2.COLOR_RGB2BGR)

            cam_overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap, 0.4, 0)

        if hook_handle is not None:
            hook_handle.remove()

    return pred_name, conf, probs.detach().cpu().numpy(), cam_overlay


def create_visual_card(img_pil, pred_name, conf, cam_overlay, img_size=128):
    orig_resized = cv2.resize(np.array(img_pil.convert("RGB")), (img_size, img_size))
    orig_bgr = cv2.cvtColor(orig_resized, cv2.COLOR_RGB2BGR)

    header_h = 36
    margin = 4
    if cam_overlay is not None:
        total_w = img_size * 2 + margin * 3
        total_h = img_size + header_h + margin * 2
        card = np.full((total_h, total_w, 3), 30, dtype=np.uint8)

        # Header color: Red for dicentric, Green for normal
        color = (30, 30, 200) if pred_name == "dicentric" else (30, 160, 30)
        cv2.rectangle(card, (0, 0), (total_w, header_h), color, -1)

        tag = f"{pred_name.upper()} ({conf*100:.1f}%)"
        cv2.putText(card, tag, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Labels for original and Grad-CAM
        y_img = header_h + margin
        x_orig = margin
        x_cam = margin * 2 + img_size
        card[y_img:y_img+img_size, x_orig:x_orig+img_size] = orig_bgr
        card[y_img:y_img+img_size, x_cam:x_cam+img_size] = cam_overlay
    else:
        total_w = img_size + margin * 2
        total_h = img_size + header_h + margin * 2
        card = np.full((total_h, total_w, 3), 30, dtype=np.uint8)

        color = (30, 30, 200) if pred_name == "dicentric" else (30, 160, 30)
        cv2.rectangle(card, (0, 0), (total_w, header_h), color, -1)

        tag = f"{pred_name[:3].upper()} {conf*100:.0f}%"
        cv2.putText(card, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        y_img = header_h + margin
        card[y_img:y_img+img_size, margin:margin+img_size] = orig_bgr

    return card


def create_mosaic_grid(cards, cols=4):
    if not cards:
        return None
    h, w, c = cards[0].shape
    rows = math.ceil(len(cards) / cols)

    grid_w = cols * w
    grid_h = rows * h
    grid = np.full((grid_h, grid_w, c), 20, dtype=np.uint8)

    for idx, card in enumerate(cards):
        r = idx // cols
        col = idx % cols
        y1, y2 = r * h, (r + 1) * h
        x1, x2 = col * w, (col + 1) * w
        grid[y1:y2, x1:x2] = card

    return grid


def main():
    args = parse_args()

    if not args.model.exists():
        print(f"[ERROR] Modelo no encontrado: {args.model}")
        sys.exit(1)

    if not args.source.exists():
        print(f"[ERROR] Origen no encontrado: {args.source}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"\n[INFO] Cargando clasificador: {args.model.name}...")
    model, arch, target_layer = load_classifier(args.model, args.arch, device)
    print(f"[INFO] Arquitectura detectada: {arch} | Dispositivo: {device}")

    # Recolectar imágenes
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if args.source.is_file():
        image_files = [args.source]
    else:
        subdirs = [d for d in args.source.iterdir() if d.is_dir()]
        if subdirs and any(d.name.lower() in ("dicentric", "normal") for d in subdirs):
            image_files = []
            per_class = max(1, args.max_samples // len(subdirs)) if args.max_samples > 0 else 999999
            for d in sorted(subdirs):
                imgs = [f for f in sorted(d.glob("*")) if f.suffix.lower() in valid_exts]
                image_files.extend(imgs[:per_class])
        else:
            image_files = [f for f in sorted(args.source.rglob("*")) if f.suffix.lower() in valid_exts]
            if args.max_samples > 0 and len(image_files) > args.max_samples:
                image_files = image_files[:args.max_samples]

    if not image_files:
        print(f"[ERROR] No se encontraron imágenes en: {args.source}")
        sys.exit(1)

    print(f"[INFO] Procesando {len(image_files)} recortes...")
    cards = []

    print("\n" + "="*70)
    print(f"{'Archivo':35s} | {'Predicción':10s} | {'Confianza':10s}")
    print("="*70)

    use_gradcam = not args.no_gradcam

    for img_path in image_files:
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Error abriendo {img_path.name}: {e}")
            continue

        pred_name, conf, probs, cam_overlay = run_inference_and_gradcam(
            model, target_layer, img_pil, args.img_size, device, use_gradcam=use_gradcam
        )

        print(f"{img_path.name[:35]:35s} | {pred_name:10s} | {conf*100:6.2f}%")

        card = create_visual_card(img_pil, pred_name, conf, cam_overlay, args.img_size)
        cards.append(card)

        # Guardar imagen individual
        out_name = f"pred_{pred_name}_{int(conf*100)}_{img_path.stem}.png"
        cv2.imwrite(str(args.output_dir / out_name), card)

    print("="*70)

    # Si se solicitó mosaico resumen
    if args.grid and cards:
        cols = 4 if len(cards) >= 4 else len(cards)
        grid_img = create_mosaic_grid(cards, cols=cols)
        grid_path = args.output_dir / "summary_grid.png"
        cv2.imwrite(str(grid_path), grid_img)
        print(f"\n[INFO] Mosaico resumen guardado en: {grid_path}")

    print(f"[INFO] Imágenes individuales guardadas en: {args.output_dir}\n")


if __name__ == "__main__":
    main()
