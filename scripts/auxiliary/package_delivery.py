"""
package_delivery.py — Empaqueta automáticamente la entrega del TFM en un archivo ZIP (< 2 GB).

Incluye:
  - scripts/ (todo el código fuente limpio sin __pycache__)
  - models/ (todos los checkpoints propios entrenados: YOLO y DenseNet121)
  - best.pt (checkpoint de YOLO raíz)
  - data/raw y data/test (conjuntos representativos de metáfases y test)
  - ResShift/ (código del modelo de difusión sin los pesos pesados de 1.1 GB)
  - results/ (métricas, CSVs, logs y carpetas de visualizaciones clave con Grad-CAM)
  - requirements.txt y README.md

Uso:
  python scripts/auxiliary/package_delivery.py
"""

import os
import shutil
import zipfile
from pathlib import Path


def get_dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    return total / (1024 * 1024)


def copy_filtered(src: Path, dst: Path, ignore_patterns=None):
    if ignore_patterns is None:
        ignore_patterns = ["__pycache__", "*.pyc", ".git", ".DS_Store"]

    def _ignore(folder, files):
        ignored = []
        for f in files:
            if f in ("__pycache__", ".git", ".pytest_cache") or f.endswith(".pyc"):
                ignored.append(f)
        return ignored

    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    elif src.is_dir():
        shutil.copytree(src, dst, ignore=_ignore, dirs_exist_ok=True)


def main():
    root = Path("C:/TFM")
    staging_dir = root / "delivery_staging" / "TFM_Entrega"
    output_zip = root / "TFM_Entrega.zip"

    print("=" * 70)
    print("EMPAQUETANDO ENTREGA DEL TFM (LÍMITE MÁXIMO: 2.00 GB)")
    print("=" * 70)

    # Limpiar staging previo si existe
    if staging_dir.parent.exists():
        shutil.rmtree(staging_dir.parent)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. Archivos raíz esenciales
    for f_name in ["README.md", "requirements.txt", "best.pt", "tfm_pipeline3.drawio"]:
        f_path = root / f_name
        if f_path.exists():
            print(f"Copiando archivo raíz: {f_name}")
            shutil.copy2(f_path, staging_dir / f_name)

    # 2. scripts/ completo
    print("Copiando carpeta: scripts/")
    copy_filtered(root / "scripts", staging_dir / "scripts")

    # 3. models/ completo (pesos propios del alumno)
    print("Copiando carpeta: models/ (pesos propios entrenados)")
    copy_filtered(root / "models", staging_dir / "models")

    # 4. data/ (solo raw y test)
    print("Copiando data/raw y data/test...")
    copy_filtered(root / "data" / "raw", staging_dir / "data" / "raw")
    copy_filtered(root / "data" / "test", staging_dir / "data" / "test")

    # 5. ResShift/ (código y configs, omitiendo weights y testdata pesados)
    print("Copiando ResShift (código fuente y configs)...")
    for sub in ["basicsr", "configs", "datapipe", "ldm", "models", "utils", "scripts"]:
        src_sub = root / "ResShift" / sub
        if src_sub.exists():
            copy_filtered(src_sub, staging_dir / "ResShift" / sub)

    for f_name in ["app.py", "inference_resshift.py", "main.py", "predict.py", "sampler.py", "trainer.py", "requirements.txt", "environment.yml", "README.md"]:
        f_path = root / "ResShift" / f_name
        if f_path.exists():
            shutil.copy2(f_path, staging_dir / "ResShift" / f_name)

    # 6. results/ (métricas consolidadas y visualizaciones clave)
    print("Copiando results (métricas y figuras clave)...")
    dst_res = staging_dir / "results"
    dst_res.mkdir(parents=True, exist_ok=True)

    # Métricas y logs directos en results/
    for f in (root / "results").glob("*"):
        if f.is_file() and f.suffix.lower() in (".json", ".csv", ".log", ".txt"):
            shutil.copy2(f, dst_res / f.name)

    # Carpetas visuales clave
    for vis_folder in [
        "classifier_visual_sr_s15",
        "classifier_visual_errors_fold1",
        "classifier_visual_val",
        "predictions_test_sr_sliced",
        "predictions_unmarked"
    ]:
        v_src = root / "results" / vis_folder
        if v_src.exists():
            copy_filtered(v_src, dst_res / vis_folder)

    # Tamaño en staging
    staging_size_mb = get_dir_size_mb(staging_dir)
    staging_size_gb = staging_size_mb / 1024
    print("\n" + "-" * 70)
    print(f"Tamaño sin comprimir del contenido seleccionado: {staging_size_mb:.2f} MB ({staging_size_gb:.2f} GB)")
    print("-" * 70)

    # 7. Crear el archivo ZIP
    print(f"\nComprimiendo en {output_zip.name}...")
    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in staging_dir.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(staging_dir.parent)
                zipf.write(file, arcname)

    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)
    zip_size_gb = zip_size_mb / 1024

    # 8. Limpiar carpeta temporal
    shutil.rmtree(staging_dir.parent)

    print("=" * 70)
    print(f"¡EMPAQUETADO COMPLETADO CON ÉXITO!")
    print(f"Archivo final: {output_zip}")
    print(f"Tamaño final comprimido: {zip_size_mb:.2f} MB ({zip_size_gb:.3f} GB)")
    if zip_size_gb <= 2.0:
        print(f"Cumple el límite de 2.00 GB (Margen libre: {2.0 - zip_size_gb:.2f} GB)")
    else:
        print(f"ALERTA: Excede el límite de 2.00 GB por {zip_size_gb - 2.0:.2f} GB")
    print("=" * 70)


if __name__ == "__main__":
    main()
