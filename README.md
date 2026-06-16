# TFM — Pipeline de Análisis de Imágenes Citogenéticas (Detección, Clasificación y Super-Resolución)

> **Trabajo de Fin de Máster en Inteligencia Artificial**
> Hardware: NVIDIA RTX 5060 (8 GB VRAM) · Python 3.10 · PyTorch 2.7+ · CUDA 12.8

---

## Descripción

Pipeline completo para el **análisis de imágenes citogenéticas** enfocado en la detección de cromosomas, clasificación de aberraciones dicéntricas y mejora de imagen mediante super-resolución por difusión (**ResShift**).

El enfoque se divide en cuatro grandes fases:
1. **Preprocesamiento:** Eliminación de artefactos circulares (OpenCV).
2. **Detección (YOLOv11):** Detección de cromosomas (1 sola clase) sobre imágenes completas y recorte (*cropping*) de cada cromosoma.
3. **Clasificación (CNN):** Clasificación individual de cada recorte en `normal` o `dicéntrico` utilizando K-Fold Cross-Validation.
4. **Superresolución (ResShift):** Aplicación de modelos de difusión sobre los recortes de los cromosomas para mejorar la calidad y posterior reevaluación del clasificador sobre los recortes mejorados.

---

## Arquitectura del Pipeline

El flujo de trabajo actual reemplaza un enfoque anterior donde YOLO evaluaba toda la imagen con dos clases y la super-resolución se aplicaba a la imagen completa. El nuevo enfoque es más eficiente y permite que el clasificador se concentre únicamente en el cromosoma, aplicando la costosa inferencia del modelo de difusión solo a los recortes pequeños.

```mermaid
graph TD
    A[Imágenes HR (Raw)] -->|OpenCV| B(Eliminación Artefactos)
    B --> C[YOLOv11 1-clase]
    C -->|Bounding Boxes| D(Cropping)
    D --> E[Crops Originales]
    
    E --> F{Clasificador CNN K-Fold}
    
    E -->|ResShift| G[Crops Super-Resueltos SR]
    G --> H{Clasificador CNN en Crops SR}
    
    F --> I((Comparación de Rendimiento Original vs SR))
    H --> I
```

---

## Estructura de Directorios (Actualizada)

```
TFM/
├── data/
│   ├── raw/
│   │   ├── unmarked/       # 50 imágenes HR originales
│   │   └── labels/         # 50 anotaciones YOLO (.txt) originales (2 clases)
│   ├── folds/              # Divisiones para entrenamiento K-Fold (Clasificador)
│   └── crops/              # Recortes extraídos de YOLO (estructurado por clases)
├── scripts/
│   ├── auxiliary/
│   │   └── preprocess.py   # Limpieza de artefactos con OpenCV
│   ├── yolo/
│   │   ├── unify_labels.py # Convierte las labels de 2 clases a 1 clase para YOLO
│   │   ├── train_yolo_1class.py # Entrena YOLOv11 con 1 clase en un split fijo
│   │   └── crop_chromosomes.py  # Inferencia YOLO, recorte de cromosomas y etiquetado IoU
│   ├── classifier/
│   │   ├── train_classifier.py  # Entrena la CNN sobre recortes con K-Fold
│   │   └── eval_classifier.py   # Evalúa la CNN sobre recortes SR vs Originales
│   ├── diffusion/
│   │   └── run_sr_crops.py      # Aplica ResShift sobre los recortes (crops)
│   └── deprecated/         # Scripts del enfoque antiguo
├── models/                 # Pesos guardados (YOLO, Clasificador, ResShift)
├── results/                # Métricas, logs y predicciones
└── README.md
```

---

## Orden de Ejecución

### Fase 1: Preparación de Datos y YOLO
1. **Unificar Etiquetas:** Convertir las etiquetas de 2 clases a 1 clase (`0` = cromosoma).
   ```powershell
   python scripts/yolo/unify_labels.py
   ```
2. **Entrenamiento de YOLO:** Entrenar YOLOv11s para detectar cromosomas (1 clase).
   ```powershell
   python scripts/yolo/train_yolo_1class.py
   ```
3. **Extracción de Recortes:** Generar los recortes de los cromosomas detectados. Las clases (0=normal, 1=dicéntrico) se asignarán automáticamente cruzando los recortes predichos con el Ground Truth original mediante IoU.
   ```powershell
   python scripts/yolo/crop_chromosomes.py
   ```

### Fase 2: Clasificación Base
4. **Entrenamiento K-Fold del Clasificador:** Entrenar la CNN en los recortes originales.
   ```powershell
   python scripts/classifier/train_classifier.py
   ```

### Fase 3: Superresolución y Reevaluación
5. **Superresolución de Recortes:** Aplicar ResShift (v1/v2/v3) a los recortes extraídos.
   ```powershell
   python scripts/diffusion/run_sr_crops.py
   ```
6. **Reevaluación del Clasificador:** Evaluar los modelos CNN (previamente entrenados en el paso 4) sobre los recortes SR y comparar resultados.
   ```powershell
   python scripts/classifier/eval_classifier.py
   ```

---

## Instalación del entorno

Se requiere PyTorch 2.7+ y CUDA 12.8 para máximo rendimiento en la RTX 5060.
```powershell
.\.venv\Scripts\Activate.ps1
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
