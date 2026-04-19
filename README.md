<<<<<<< HEAD
# TFM — Super-Resolución de Imágenes Citogenéticas con Modelos de Difusión + YOLOv11

> **Trabajo de Fin de Máster**  
> Hardware objetivo: NVIDIA RTX 5060 (8 GB VRAM)  
> Python 3.10 | Entorno virtual `.venv`

---

## Descripción

Este proyecto implementa un pipeline completo para la **super-resolución de imágenes citogenéticas** usando modelos de difusión (**StableSR**) y evalúa el impacto de la super-resolución en la detección automática de cromosomas con **YOLOv11**.

El dataset consta de 50 imágenes de alta resolución de cariotipo humano. El pipeline completo incluye:
1. Preprocesamiento y normalización
2. Data augmentation consistente (imagen + bounding boxes)
3. Simulación de baja resolución (LR)
4. Baseline de detección YOLO (validación cruzada K-Fold)
5. Finetuning de StableSR sobre el dataset citogenético
6. Inferencia SR y evaluación cuantitativa (PSNR, SSIM, LPIPS, mAP)

---

## Estructura de directorios

```
TFM/
├── data/
│   ├── raw/            # 50 imágenes originales (no modificar)
│   ├── processed/      # Imágenes preprocesadas (referencia HR)
│   ├── lr/x2|x3|x4/   # Imágenes LR simuladas
│   ├── sr/x2|x3|x4/   # Resultados de super-resolución
│   └── annotations/    # Anotaciones YOLO (.txt)
├── scripts/            # Scripts Python del pipeline
├── configs/            # Configuraciones YAML (YOLO + StableSR)
├── results/
│   ├── metrics/        # CSV con PSNR, SSIM, mAP, etc.
│   ├── plots/          # Gráficas comparativas
│   └── logs/           # Logs de ejecución
├── models/
│   ├── yolo_baseline/  # Pesos YOLO entrenados
│   └── stablesr_finetuned/ # Checkpoints StableSR
├── .venv/              # Entorno virtual Python
├── requirements.txt
└── README.md
```

---

## Instalación del entorno

```powershell
# Activar el entorno virtual (desde C:\TFM)
.\.venv\Scripts\Activate.ps1

# Verificar instalación
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Orden de ejecución

### Fase 1 — Setup y datos

```bash
# Preprocesar imágenes raw
python scripts/00_preprocess.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --target_size 512 \
    --convert_gray

# Generar imágenes LR x2
python scripts/02_degrade.py \
    --input_dir data/processed \
    --output_dir data/lr/x2 \
    --scale 2 \
    --degradation blur_noise
```

### Fase 2 — Baseline YOLO

```bash
python scripts/03_train_yolo_baseline.py \
    --data_dir data/processed \
    --annot_dir data/annotations \
    --output_dir models/yolo_baseline \
    --model_size n \
    --cv_mode kfold \
    --k_folds 5 \
    --epochs 50 \
    --exp_name baseline_kfold5
```

### Fase 3 — Finetuning StableSR (repetir por experimento)

```bash
python scripts/04_train_stablesr.py \
    --config configs/stablesr_default.yaml \
    --hr_dir data/processed \
    --lr_dir data/lr/x2 \
    --output_dir models/stablesr_finetuned \
    --exp_name exp0_default
```

### Fase 4 — Inferencia SR

```bash
python scripts/05_run_sr.py \
    --lr_dir data/lr/x2 \
    --output_dir data/sr/x2 \
    --checkpoint models/stablesr_finetuned/exp0_default/checkpoints/best.ckpt \
    --scale 2 \
    --exp_name exp0_default
```

### Fase 5 — Evaluación

```bash
# Métricas de imagen (PSNR, SSIM)
python scripts/07_compute_metrics.py \
    --sr_dir data/sr/x2/exp0_default \
    --hr_dir data/processed \
    --output_csv results/metrics/sr_exp0_default.csv \
    --exp_name exp0_default

# Evaluación YOLO sobre SR
python scripts/06_evaluate_yolo.py \
    --sr_dir data/sr/x2/exp0_default \
    --annot_dir data/annotations \
    --yolo_checkpoint models/yolo_baseline/best.pt \
    --output_dir results/metrics \
    --exp_name yolo_on_sr_exp0
```

---

## Experimentos StableSR

| Experimento | Scheduler | Steps | Loss     | Freeze UNet | Notas               |
|-------------|-----------|-------|----------|-------------|---------------------|
| exp0        | DDIM      | 20    | L1+LPIPS | ✓           | Configuración base  |
| exp1        | DDPM      | 50    | L1+LPIPS | ✓           | Más pasos denoising |
| exp2        | DDIM      | 20    | L2       | ✓           | Pérdida MSE         |
| exp3        | DDIM      | 20    | L1+LPIPS | ✗           | Finetuning completo |
| exp4 (x4)  | DDIM      | 20    | L1+LPIPS | ✓           | Factor x4 agresivo  |

---

## Convenciones de código

- Todos los scripts son ejecutables por CLI (`argparse`), sin paths hardcodeados.
- `--exp_name` identifica cada experimento en logs, checkpoints y CSVs.
- `--seed 42` para reproducibilidad (PyTorch + NumPy + random).
- `pathlib.Path` para manejo de rutas.
- `tqdm` en todos los bucles de imágenes y épocas.
- `logging` a consola + `results/logs/<exp_name>.log`.
- Gestión de errores suave: warning + continue (sin crash).
- Verificación de CUDA al inicio de cada script.
=======
# tfm-citogenetica
>>>>>>> e44a874129092de689c1905920731915e84b6b7c
