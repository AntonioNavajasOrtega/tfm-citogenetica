# TFM — Super-resolución de imágenes citogenéticas mediante modelos de difusión

> **Trabajo de Fin de MÁSTER UNIVERSITARIO EN INGENIERÍA DEL SOFTWARE E INTELIGENCIA ARTIFICIAL — Universidad de Málaga (UMA)**
> Hardware: NVIDIA RTX 5060 (8 GB VRAM) · Python 3.10 · PyTorch 2.7+ · CUDA 12.8

## 📄 Descripción

Este Trabajo de Fin de Máster estudia si la **super-resolución basada en modelos de difusión** mejora el análisis automático de cromosomas dicéntricos en imágenes citogenéticas.

Para ello se desarrolla un pipeline que combina:
- **YOLOv11s** para la detección y extracción de cromosomas individuales.
- **ResShift** para la generación de imágenes super-resueltas (×4).
- **DenseNet121** para la clasificación binaria entre cromosomas normales y dicéntricos.

Se comparan imágenes de alta resolución (HR) originales con sus versiones super-resueltas (SR), considerando tanto el modelo ResShift preentrenado como una versión ajustada (*fine-tuned*) al dominio citogenético.

> **Resultado principal:** La super-resolución no mejora el rendimiento de detección ni de clasificación respecto al uso directo de imágenes HR. Una mejora en la calidad perceptual de la imagen no implica necesariamente una mejora en las tareas diagnósticas posteriores.

El conjunto de datos es **privado** (50 imágenes de metáfase en blanco y negro, proporcionadas por el grupo ICAI en colaboración con el Dr. Adayabalam S. Balajee, REAC/TS, ORISE, Tennessee) e irradiadas a 2 Gy.


## 📊 Resultados Principales

### Detección — YOLOv11s (1 clase, K-Fold de 5 pliegues, imgsz=1280)

| Métrica     | Media ± Desv.       |
|-------------|---------------------|
| mAP@50      | 98.6% ± 0.4%        |
| mAP@50-95   | 85.5% ± 0.5%        |
| Precision   | 97.9% ± 0.7%        |
| Recall      | 97.7% ± 0.6%        |
| F1-Score    | 97.8% ± 0.2%        |

> Sobre imágenes HR originales. Sobre teselas SR el mismo modelo obtiene F1 macro de 0.0195 por domain shift severo.

### Clasificación de Dicéntricos — DenseNet121 (K-Fold de 5 pliegues)

| Variante de entrada       | F1 dicéntrico (media ± desv.) | Accuracy global   |
|---------------------------|-------------------------------|-------------------|
| **HR (original)**         | **0.7985 ± 0.0798**           | 0.9763 ± 0.0109   |
| SR — ResShift 15 pasos    | 0.6878 ± 0.0487               | 0.9660 ± 0.0032   |
| SR — ResShift 4 pasos     | 0.6551 ± 0.0994               | 0.9590 ± 0.0173   |
| SR — ResShift 15p + FT    | 0.4525 ± 0.0536               | 0.9029 ± 0.0059   |

> La accuracy global es poco informativa con ~5% de dicéntricos por fold. El F1 de dicéntrico es la métrica relevante. **HR supera a todas las variantes SR.** El fine-tuning mejora la calidad perceptual (PSNR, LPIPS) pero empeora la clasificación.

---

## 📂 Estructura de Directorios

```
TFM/
├── data/
│   ├── raw/
│   │   ├── unmarked/                   # 50 imágenes HR originales (privadas)
│   │   └── labels/                     # Anotaciones YOLO — 2 clases originales
│   ├── folds/                          # Particiones K-Fold para clasificación
│   ├── crops/                          # Recortes extraídos por YOLO (por clase)
│   └── test/                           # Conjunto de test con parches SR
├── scripts/
│   ├── auxiliary/
│   │   └── preprocess.py               # Limpieza de artefactos circulares (OpenCV)
│   ├── yolo/
│   │   ├── unify_labels.py             # Convierte etiquetas 2 clases → 1 clase
│   │   ├── train_yolo_1class.py        # Entrena YOLOv11s — 1 clase, K-Fold
│   │   ├── crop_chromosomes.py         # Inferencia YOLO + recorte + etiquetado IoU
│   │   └── run_test_slices_predict.py  # Inferencia sobre parches SR de test
│   ├── classifier/
│   │   ├── train_classifier.py         # Entrena DenseNet121 con K-Fold
│   │   ├── eval_classifier.py          # Evalúa CNN sobre crops SR vs HR
│   │   └── predict_visual.py           # Inferencia + mapas de calor Grad-CAM
│   ├── diffusion/
│   │   └── run_sr_crops.py             # Aplica ResShift (×4) sobre los recortes
│   └── deprecated/                     # Scripts del enfoque 1 (SR sobre imagen completa)
├── models/
│   ├── yolo_single/                    # Pesos YOLOv11s entrenados (1 clase)
│   ├── classifier/                     # Pesos DenseNet121 — crops HR (K-Fold)
│   ├── classifier_sr/                  # Pesos DenseNet121 — crops SR (K-Fold)
│   └── resshift_finetuned/             # Pesos ResShift ajustados al dominio
├── results/
│   ├── metrics/                        # CSVs con métricas por fold y comparativa
│   ├── logs/                           # Logs de inferencia ResShift por fold
│   ├── predictions_test_sr_sliced/     # Predicciones YOLO sobre test SR
│   └── eval_visual_test/               # Tarjetas de predicción y Grad-CAM
├── ResShift/                           # Submódulo oficial de ResShift (terceros)
├── requirements.txt                    # Dependencias del pipeline principal
├── requirementsResshift.txt            # Dependencias específicas de ResShift
└── README.md
```

---

## ⚙️ Instalación del Entorno

ResShift requiere un entorno separado por incompatibilidades de dependencias con el pipeline principal.

### Entorno principal (YOLO + Clasificador)

```powershell
conda create -n env_main python=3.10 -y
conda activate env_main
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Entorno ResShift (superresolución)

```powershell
conda create -n env_resshift python=3.10 -y
conda activate env_resshift
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirementsResshift.txt
```

Los pesos preentrenados de ResShift (`resshift_realsrx4_s15_v1.pth`, ~1.1 GB) deben descargarse desde el [repositorio oficial de ResShift](https://github.com/zsyOAOA/ResShift/releases) y colocarse en `ResShift/weights/`.

> **Nota sobre compatibilidad con RTX 5060 (Blackwell, sm_120):** El código original de ResShift no es compatible con esta arquitectura. Se aplicaron modificaciones en `ldm/modules/diffusionmodules/model.py` para propagar el parámetro `padding_mode` en las capas convolucionales del autoencoder y la U-Net de difusión.

---

## 🚀 Orden de Ejecución

### Fase 1 — Detección y extracción de recortes (entorno `env_main`)

**1. Preprocesamiento:** Eliminar artefactos circulares de las imágenes originales.
```powershell
python scripts/auxiliary/preprocess.py
```

**2. Unificar etiquetas:** Convertir las anotaciones de 2 clases a 1 clase (`0` = cromosoma).
```powershell
python scripts/yolo/unify_labels.py
```

**3. Entrenar YOLOv11s** con validación cruzada K-Fold (imgsz=1280).
```powershell
python scripts/yolo/train_yolo_1class.py
```

**4. Extraer recortes:** Inferencia YOLO + cropping + asignación de clase por IoU con el Ground Truth.
```powershell
python scripts/yolo/crop_chromosomes.py
```

### Fase 2 — Clasificación base sobre crops HR (entorno `env_main`)

**5. Entrenar clasificador K-Fold:** DenseNet121 sobre los recortes originales (HR).
```powershell
python scripts/classifier/train_classifier.py
```

### Fase 3 — Superresolución y reevaluación (entorno `env_resshift`)

**6. Aplicar ResShift ×4** sobre los recortes extraídos.
```powershell
python scripts/diffusion/run_sr_crops.py
```

**7. Reevaluar clasificador:** Evaluar los modelos DenseNet121 sobre los crops SR.
```powershell
# Volver al entorno principal
conda activate env_main
python scripts/classifier/eval_classifier.py
```

---

## 🔍 Reproducción Rápida (para el Evaluador)

Los modelos entrenados están incluidos en `models/`. Para verificar el funcionamiento sin reentrenar:

**1. Inferencia YOLO sobre metáfases SR (conjunto de test):**
```powershell
python scripts/yolo/run_test_slices_predict.py
```
*Las imágenes con detecciones se guardan en `results/predictions_test_sr_sliced/`.*

**2. Clasificación + mapas de calor Grad-CAM (DenseNet121):**
```powershell
python scripts/classifier/predict_visual.py `
    --model models/classifier/cnn_sr_s15_fold0.pth `
    --source data/test/test_sr_sliced_1280/images `
    --output_dir results/test_gradcam `
    --max_samples 8 --grid
```
*Genera tarjetas de predicción individuales y el mosaico `summary_grid.png` en `results/test_gradcam/`.*

**3. Evaluación completa con métricas por fold:**
```powershell
python scripts/classifier/eval_classifier.py `
    --model_path models/classifier_sr/cnn_densenet_folds_sr_fold0.pth `
    --data_dir data/test/test_sr_sliced_1280/images `
    --save_visual results/eval_visual
```

---

## 📎 Nota sobre Datos y Pesos

- **Conjunto de datos:** Las 50 metáfases HR con etiquetas originales se incluyen en `data/raw/`. El conjunto de test con parches SR se incluye en `data/test/`. Los recortes intermedios de todos los folds (~6 GB) están disponibles mediante el enlace especificado en la memoria del TFM (límite de repositorio institucional de 2 GB).
- **Pesos de ResShift (terceros):** Los pesos base preentrenados se descargan desde el [repositorio oficial de ResShift](https://github.com/zsyOAOA/ResShift/releases) y se colocan en `ResShift/weights/`. Todos los modelos propios entrenados (YOLOv11s y DenseNet121) se incluyen íntegros en `models/`.
- **StableSR fue descartado** por incompatibilidad con la RTX 5060 (Blackwell, `sm_120`) y por requerir al menos 24 GB de VRAM para su fine-tuning.

---

## 👤 Créditos

| Campo       | Información                                         |
|-------------|-----------------------------------------------------|
| **Autor**   | Antonio Navajas Ortega                              |
| **Tutor**   | Ezequiel López Rubio                                |
| **Cotutor** | Miguel Ángel Molina Cabello                         |
| **Máster**  | Ingeniería de Software e Inteligencia Artificial    |
| **Centro**  | Universidad de Málaga — ETSI Informática            |
| **Fecha**   | Septiembre 2026                                     |

---

## 📑 Palabras Clave

Super-resolución de imágenes · Imágenes citogenéticas · Aprendizaje profundo · Modelos de difusión · Visión por computador · Cromosomas dicéntricos · ResShift · YOLOv11 · DenseNet
