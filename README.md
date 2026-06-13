
# TFM — Super-Resolución de Imágenes Citogenéticas con Modelos de Difusión + YOLOv11

> **Trabajo de Fin de Máster**
> Hardware: NVIDIA RTX 5060 (8 GB VRAM) · Python 3.10

---

## Descripción

Pipeline completo para la **super-resolución de imágenes citogenéticas** mediante modelos de difusión (**StableSR**) y evaluación del impacto en la detección automática de cromosomas y aberraciones con **YOLOv11**.

El dataset consta de **50 imágenes HR de cariotipo humano** (dosis 2 Gy) con **anotaciones YOLO de dominio experto** en dos clases:
- clase `0` — cromosoma normal
- clase `1` — aberración dicéntrica

El objetivo es demostrar que la super-resolución por difusión mejora la detección de cromosomas/aberraciones frente al upsampling clásico (bicúbico).

---

## Estructura de directorios

```
TFM/
├── data/
│   ├── raw/
│   │   ├── unmarked/       # 50 imágenes HR originales (no tocar)
│   │   ├── marked/         # versiones con marcas visuales (referencia)
│   │   └── labels/         # 50 anotaciones YOLO (.txt) — fuente única
│   ├── processed/          # imágenes preprocesadas 512×512 (referencia HR)
│   ├── augmented/
│   │   ├── images/         # imágenes aumentadas (~150)
│   │   └── labels/         # bboxes transformadas junto a las imágenes
│   ├── lr/x2/              # imágenes LR simuladas 256×256
│   └── sr/x2/              # resultados SR organizados por experimento
│       ├── exp0_default/
│       ├── exp1_ddpm50/
│       └── exp2_l2loss/
├── scripts/
│   ├── preprocess.py       # raw → processed (redimensionar, normalizar)
│   ├── augment.py          # processed → augmented (imagen + bboxes)
│   ├── degrade.py          # processed → lr/x2 (simular baja resolución)
│   ├── train_yolo.py       # entrenar yolo con k-fold sobre processed
│   ├── diffusion/          # scripts integrados para ResShift y StableSR
│   │   ├── train_resshift.py # finetuning de ResShift (hr + lr → checkpoint)
│   │   ├── run_resshift.py   # inferencia sr nativa con ResShift
│   │   └── run_stablesr.py   # inferencia sr delegada a StableSR
│   ├── evaluate_yolo.py    # yolo sobre imágenes sr (mAP, F1)
│   └── compute_metrics.py  # métricas de imagen sr vs hr (psnr, ssim)
├── configs/
│   ├── stablesr_default.yaml   # exp0: DDIM 20 steps, L1+LPIPS
│   ├── stablesr_exp1.yaml      # exp1: DDPM 50 steps, L1+LPIPS
│   ├── stablesr_exp2.yaml      # exp2: DDIM 20 steps, L2
│   └── yolo_dataset.yaml
├── models/
│   ├── yolo_baseline/          # pesos YOLO por fold
│   └── stablesr_finetuned/     # checkpoints StableSR por experimento
├── results/
│   ├── metrics/                # CSV con PSNR, SSIM, mAP, F1
│   ├── plots/                  # gráficas comparativas
│   └── logs/                   # logs de ejecución por script/experimento
├── .venv/
├── requirements.txt
└── README.md
```

> **Nota sobre anotaciones**: el único directorio de etiquetas es `data/raw/labels/`.
> Las imágenes aumentadas generan sus propias etiquetas transformadas en `data/augmented/labels/`.

---

## Instalación del entorno

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Estado del pipeline

| Paso | Script | Estado |
|---|---|---|
| 1. Preprocesamiento | `preprocess.py` | ✅ ejecutado — 50 imgs en `data/processed/` |
| 2. Degradación LR | `degrade.py` | ✅ ejecutado — 50 imgs en `data/lr/x2/` |
| 3. Data augmentation | `augment.py` | ⏳ pendiente |
| 4. Entrenamiento YOLO baseline | `train_yolo.py` | ⏳ pendiente |
| 5. Entrenamiento ResShift | `diffusion/train_resshift.py` | ⏳ pendiente |
| 6. Inferencia SR | `diffusion/run_*.py` | ⏳ pendiente |
| 7. Evaluación YOLO sobre SR | `evaluate_yolo.py` | ⏳ pendiente |
| 8. Métricas PSNR/SSIM | `compute_metrics.py` | ⏳ pendiente |

---

## Orden de ejecución

### Fase 1 — Preparar datos aumentados

```powershell
python scripts/augment.py `
    --input_dir data/processed `
    --annot_dir data/raw/labels `
    --output_dir data/augmented/images `
    --output_annot data/augmented/labels `
    --factor 3 --seed 42
```

### Fase 2 — Baseline YOLO (sobre imágenes HR originales)

```powershell
python scripts/train_yolo.py `
    --data_dir data/processed `
    --annot_dir data/raw/labels `
    --output_dir models/yolo_baseline `
    --model_size n --cv_mode kfold --k_folds 5 `
    --epochs 50 --exp_name baseline_kfold5
```

### Fase 3 — Entrenamiento de Modelos de Difusión

```powershell
# Ejemplo: Finetuning de ResShift
python scripts/diffusion/train_resshift.py `
    --config ResShift/configs/realsr_swinunet_realesrgan256.yaml `
    --hr_dir data/processed --lr_dir data/lr/x2 `
    --exp_name resshift_exp1_finetune --epochs 50
```

### Fase 4 — Inferencia SR (repetir por experimento)

```powershell
# Inferencia con StableSR
python scripts/diffusion/run_stablesr.py `
    --input_dir data/lr/x2 `
    --output_dir data/sr/x2 `
    --sampler ddim --ddpm_steps 50 --colorfix wavelet `
    --exp_name stablesr_ddim_50

# Inferencia con ResShift
python scripts/diffusion/run_resshift.py `
    --input_dir data/lr/x2 `
    --output_dir data/sr/x2 `
    --checkpoint ResShift/weights/resshift_realsrx4_s15_v1.pth `
    --task realsr --scale 2 --exp_name resshift_v1_eval
```

### Fase 5 — Evaluación

```powershell
# métricas de imagen: SR vs HR original
python scripts/compute_metrics.py `
    --sr_dir data/sr/x2/exp0_default `
    --hr_dir data/processed `
    --metrics psnr ssim `
    --output_csv results/metrics/sr_exp0_default.csv `
    --exp_name exp0_default

# detección YOLO sobre imágenes SR
python scripts/evaluate_yolo.py `
    --sr_dir data/sr/x2/exp0_default `
    --annot_dir data/raw/labels `
    --yolo_checkpoint models/yolo_baseline/baseline_kfold5_fold1/train/weights/best.pt `
    --output_dir results/metrics `
    --exp_name yolo_on_sr_exp0
```

---

## Diseño de experimentos

### Bloque A — Impacto de SR en detección (mAP50, F1)

| Exp | Entrada YOLO | Descripción |
|---|---|---|
| A0 | HR original 512×512 | techo de rendimiento — referencia |
| A1 | LR x2 upscaled bicúbico | cuánto pierde la detección sin SR |
| A2 | SR exp0 (DDIM 20, L1+LPIPS) | SR base vs. bicúbic |
| A3 | SR exp1 (DDPM 50, L1+LPIPS) | más pasos de denoising |
| A4 | SR exp2 (DDIM 20, L2) | función de pérdida alternativa |

### Bloque B — Calidad de imagen SR (PSNR, SSIM)

| Exp | Método | Descripción |
|---|---|---|
| B0 | Bicúbico | baseline sin aprendizaje |
| B1 | StableSR exp0 | configuración estándar |
| B2 | StableSR exp1 | DDPM / más pasos |
| B3 | StableSR exp2 | pérdida L2 |

### Bloque C — Ablación de degradación

| Exp | Degradación | Escala |
|---|---|---|
| C0 | bicubic | x2 |
| C1 | blur + noise | x2 |

---

## Convenciones de código

- scripts ejecutables por CLI (`argparse`), sin rutas hardcodeadas
- `--exp_name` identifica cada experimento en logs, checkpoints y CSVs
- `--seed 42` para reproducibilidad (PyTorch + NumPy + random)
- `pathlib.Path` para rutas (compatible Windows/Linux)
- `tqdm` en todos los bucles de imágenes y épocas
- `logging` a consola + `results/logs/<exp_name>.log`
- gestión de errores suave: warning + continue (sin crash)
- verificación de CUDA al inicio de cada script
