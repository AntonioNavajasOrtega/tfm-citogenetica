"""
run_sr_pipeline.py — Orquestador completo del pipeline de superresolución.

Encadena en orden:
  1. StableSR  sobre LR x2 y LR x4          (reconstrucción)
  2. StableSR  sobre HR originales            (upscale x2)
  3. ResShift pretrained sobre LR x2 y LR x4  (reconstrucción)
  4. ResShift pretrained sobre HR originales   (upscale x4)
  5. Finetune ResShift x2 sobre el dataset propio
  6. Finetune ResShift x4 sobre el dataset propio
  7. ResShift finetuned sobre LR x2 y LR x4   (reconstrucción)
  8. Cálculo de métricas PSNR/SSIM/LPIPS para cada experimento SR
  9. Evaluación YOLO downstream (baseline + cada set SR de test)
 10. Tabla comparativa final

Uso completo (todas las fases):
    python scripts/diffusion/run_sr_pipeline.py

Saltar el finetune (solo inferencia con preentrenado):
    python scripts/diffusion/run_sr_pipeline.py --skip_train

Solo métricas + YOLO (si las imágenes SR ya existen):
    python scripts/diffusion/run_sr_pipeline.py --skip_sr --skip_train

Fases individuales:
    python scripts/diffusion/run_sr_pipeline.py --phases stablesr metrics yolo
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Constantes de rutas
# ─────────────────────────────────────────────────────────────────────────────
ROOT        = Path("c:/TFM").resolve()
SCRIPTS_SR  = ROOT / "scripts/diffusion"
SCRIPTS_YOLO= ROOT / "scripts/yolo"

# Datos
LR_X2   = ROOT / "data/lr/x2"
LR_X4   = ROOT / "data/lr/x4"
HR_DIR  = ROOT / "data/raw/unmarked"
TEST_IMG= ROOT / "data/test/images"
TEST_LBL= ROOT / "data/test/labels"

# Checkpoints
STABLESR_CKPT = ROOT / "StableSR/checkpoints/stablesr_768v_000139.ckpt"
VQGAN_CKPT    = ROOT / "StableSR/checkpoints/vqgan_cfw_00011.ckpt"
YOLO_CKPT     = ROOT / "runs/detect/runs/experimentos/yolo11s_img1280/yolo11s_img1280_fold4/train/weights/best.pt"

# Salida
SR_DIR      = ROOT / "results/sr"
METRICS_DIR = ROOT / "results/metrics"

# ─────────────────────────────────────────────────────────────────────────────
# Definición de experimentos
# ─────────────────────────────────────────────────────────────────────────────

# Cada entrada: (exp_name, input_dir, scale, model, extra_kwargs)
# model ∈ {"stablesr", "resshift_pretrained", "resshift_finetuned"}
SR_EXPERIMENTS = [
    # ── StableSR ─────────────────────────────────────────────────────────────
    {
        "exp_name": "stablesr_x2_recon",
        "input_dir": LR_X2,
        "scale": 2,
        "model": "stablesr",
        "hr_ref": HR_DIR,        # referencia para métricas
    },
    {
        "exp_name": "stablesr_x4_recon",
        "input_dir": LR_X4,
        "scale": 4,
        "model": "stablesr",
        "hr_ref": HR_DIR,
    },
    {
        "exp_name": "stablesr_x2_upscale",
        "input_dir": HR_DIR,
        "scale": 2,
        "model": "stablesr",
        "hr_ref": None,          # sin referencia (SR por encima de la original)
    },
    # ── ResShift pretrained ───────────────────────────────────────────────────
    {
        "exp_name": "resshift_pretrained_x2_recon",
        "input_dir": LR_X2,
        "scale": 2,
        "model": "resshift_pretrained",
        "version": "v3",
        "hr_ref": HR_DIR,
    },
    {
        "exp_name": "resshift_pretrained_x4_recon",
        "input_dir": LR_X4,
        "scale": 4,
        "model": "resshift_pretrained",
        "version": "v3",
        "hr_ref": HR_DIR,
    },
    {
        "exp_name": "resshift_pretrained_x4_upscale",
        "input_dir": HR_DIR,
        "scale": 4,
        "model": "resshift_pretrained",
        "version": "v3",
        "hr_ref": None,
    },
    # ── ResShift finetuned ────────────────────────────────────────────────────
    {
        "exp_name": "resshift_finetuned_x2_recon",
        "input_dir": LR_X2,
        "scale": 2,
        "model": "resshift_finetuned",
        "finetune_exp": "resshift_finetune_x2",
        "hr_ref": HR_DIR,
    },
    {
        "exp_name": "resshift_finetuned_x4_recon",
        "input_dir": LR_X4,
        "scale": 4,
        "model": "resshift_finetuned",
        "finetune_exp": "resshift_finetune_x4",
        "hr_ref": HR_DIR,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path, level: str = "INFO") -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución de subcomandos
# ─────────────────────────────────────────────────────────────────────────────

PYTHON_STABLESR = [sys.executable]
PYTHON_RESSHIFT = [sys.executable]

def run(cmd: list[str], label: str, logger: logging.Logger, cwd: Path = ROOT) -> bool:
    """Lanza un subproceso y devuelve True si tuvo éxito."""
    logger.info(f"\n{'='*60}")
    logger.info(f"▶  {label}")
    logger.info(f"   {' '.join(str(c) for c in cmd)}")
    logger.info(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        logger.error(f"X  FALLÓ: {label}  (código {result.returncode})")
        return False
    logger.info(f"OK: {label}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Fases del pipeline
# ─────────────────────────────────────────────────────────────────────────────

def phase_stablesr(experiments: list[dict], logger: logging.Logger) -> list[str]:
    """Lanza StableSR para todos los experimentos de tipo 'stablesr'. Devuelve lista de exp_name procesados."""
    done = []
    for exp in experiments:
        if exp["model"] != "stablesr":
            continue
        exp_name = exp["exp_name"]
        out_dir  = SR_DIR / exp_name

        # Si ya existe con imágenes, saltar
        if out_dir.exists() and any(out_dir.iterdir()):
            logger.info(f"  [SKIP] {exp_name} — ya existe en {out_dir}")
            done.append(exp_name)
            continue

        ok = run(
            cmd=[
                *PYTHON_STABLESR, str(SCRIPTS_SR / "run_stablesr.py"),
                "--input_dir",  str(exp["input_dir"]),
                "--output_dir", str(SR_DIR),
                "--checkpoint", str(STABLESR_CKPT),
                "--vqgan_ckpt", str(VQGAN_CKPT),
                "--scale",      str(exp["scale"]),
                "--exp_name",   exp_name,
            ],
            label=f"StableSR — {exp_name}",
            logger=logger,
        )
        if ok:
            done.append(exp_name)
    return done


def phase_resshift_pretrained(experiments: list[dict], logger: logging.Logger) -> list[str]:
    """Lanza ResShift con pesos preentrenados (se descargan automáticamente)."""
    done = []
    for exp in experiments:
        if exp["model"] != "resshift_pretrained":
            continue
        exp_name = exp["exp_name"]
        out_dir  = SR_DIR / exp_name

        if out_dir.exists() and any(out_dir.iterdir()):
            logger.info(f"  [SKIP] {exp_name} — ya existe en {out_dir}")
            done.append(exp_name)
            continue

        ok = run(
            cmd=[
                *PYTHON_RESSHIFT, str(SCRIPTS_SR / "run_resshift.py"),
                "--input_dir",  str(exp["input_dir"]),
                "--output_dir", str(SR_DIR),
                "--scale",      str(exp["scale"]),
                "--version",    exp.get("version", "v3"),
                "--exp_name",   exp_name,
            ],
            label=f"ResShift pretrained — {exp_name}",
            logger=logger,
        )
        if ok:
            done.append(exp_name)
    return done


def phase_resshift_finetune(scales: list[int], iterations: int,
                             logger: logging.Logger) -> dict[int, Path]:
    """
    Lanza el finetune de ResShift para cada escala.
    Devuelve dict {scale: ruta_al_best_ckpt}.
    """
    ckpt_map: dict[int, Path] = {}
    for scale in scales:
        exp_name = f"resshift_finetune_x{scale}"
        lr_dir   = LR_X2 if scale == 2 else LR_X4
        out_dir  = ROOT / "models/resshift_finetuned" / exp_name

        # Buscar si ya hay checkpoints
        ckpt_candidates = list(out_dir.glob("ckpts/*.pth")) if out_dir.exists() else []
        if ckpt_candidates:
            best = max(ckpt_candidates, key=lambda p: p.stat().st_mtime)
            logger.info(f"  [SKIP] finetune x{scale} — usando checkpoint: {best}")
            ckpt_map[scale] = best
            continue

        ok = run(
            cmd=[
                *PYTHON_RESSHIFT, str(SCRIPTS_SR / "train_resshift.py"),
                "--hr_dir",     str(HR_DIR),
                "--lr_dir",     str(lr_dir),
                "--scale",      str(scale),
                "--iterations", str(iterations),
                "--batch_size", "2",
                "--save_freq",  str(max(500, iterations // 10)),
                "--exp_name",   exp_name,
            ],
            label=f"ResShift finetune x{scale}",
            logger=logger,
        )
        if ok:
            ckpt_candidates = list((out_dir / "ckpts").glob("*.pth")) if (out_dir / "ckpts").exists() else []
            if ckpt_candidates:
                best = max(ckpt_candidates, key=lambda p: p.stat().st_mtime)
                ckpt_map[scale] = best
    return ckpt_map


def phase_resshift_finetuned(experiments: list[dict], ckpt_map: dict[int, Path],
                              logger: logging.Logger) -> list[str]:
    """Inferencia con los pesos finetuneados."""
    done = []
    for exp in experiments:
        if exp["model"] != "resshift_finetuned":
            continue
        exp_name = exp["exp_name"]
        scale    = exp["scale"]

        if scale not in ckpt_map:
            logger.warning(f"  [SKIP] {exp_name} — no hay checkpoint finetuneado x{scale}")
            continue

        out_dir = SR_DIR / exp_name
        if out_dir.exists() and any(out_dir.iterdir()):
            logger.info(f"  [SKIP] {exp_name} — ya existe en {out_dir}")
            done.append(exp_name)
            continue

        ok = run(
            cmd=[
                *PYTHON_RESSHIFT, str(SCRIPTS_SR / "run_resshift.py"),
                "--input_dir",  str(exp["input_dir"]),
                "--output_dir", str(SR_DIR),
                "--scale",      str(scale),
                "--checkpoint", str(ckpt_map[scale]),
                "--exp_name",   exp_name,
            ],
            label=f"ResShift finetuned — {exp_name}",
            logger=logger,
        )
        if ok:
            done.append(exp_name)
    return done


def phase_metrics(experiments: list[dict], done_sr: set[str],
                  logger: logging.Logger) -> list[str]:
    """Calcula PSNR/SSIM/LPIPS para cada experimento SR que tenga referencia HR."""
    computed = []
    for exp in experiments:
        exp_name = exp["exp_name"]
        hr_ref   = exp.get("hr_ref")

        if exp_name not in done_sr:
            logger.info(f"  [SKIP métricas] {exp_name} — no se generó SR")
            continue
        if hr_ref is None:
            logger.info(f"  [SKIP métricas] {exp_name} — sin referencia HR (upscale)")
            continue

        sr_dir     = SR_DIR / exp_name
        csv_out    = METRICS_DIR / f"{exp_name}.csv"

        ok = run(
            cmd=[
                sys.executable, str(SCRIPTS_SR / "compute_metrics.py"),
                "--sr_dir",     str(sr_dir),
                "--hr_dir",     str(hr_ref),
                "--output_csv", str(csv_out),
                "--metrics",    "psnr", "ssim", "lpips",
                "--exp_name",   exp_name,
            ],
            label=f"Métricas — {exp_name}",
            logger=logger,
        )
        if ok:
            computed.append(exp_name)
    return computed


def phase_yolo(done_sr: set[str], logger: logging.Logger) -> None:
    """
    Evalúa YOLO sobre:
      - Baseline: imágenes de test originales
      - SR reconstructed: LR de test reconstruidas con cada modelo
    
    Nota: para evaluar SR en el conjunto de test necesitamos primero generar
    las LR de test (si no existen) y luego aplicar SR sobre ellas.
    El pipeline evalúa directamente las imágenes SR que correspondan a nombres
    del conjunto de test (buscando coincidencia de nombres en la carpeta SR).
    """

    def _eval(img_dir: Path, exp_name: str):
        csv_out = METRICS_DIR / "yolo_comparison.csv"
        run(
            cmd=[
                sys.executable, str(SCRIPTS_YOLO / "evaluate_yolo.py"),
                "--img_dir",    str(img_dir),
                "--lbl_dir",    str(TEST_LBL),
                "--checkpoint", str(YOLO_CKPT),
                "--img_size",   "1280",
                "--exp_name",   exp_name,
                "--output_dir", str(METRICS_DIR),
            ],
            label=f"YOLO eval — {exp_name}",
            logger=logger,
        )

    # 1) Baseline: imágenes de test originales (sin SR)
    _eval(TEST_IMG, "baseline_no_sr")

    # 2) SR sobre imágenes de test
    #    Para cada experimento de reconstrucción, comprobamos si hay imágenes
    #    en results/sr/<exp_name>/ cuyos nombres coincidan con los del test set
    test_names = {p.stem for p in TEST_IMG.iterdir()
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif"}}

    for exp_name in done_sr:
        sr_dir = SR_DIR / exp_name
        sr_names = {p.stem for p in sr_dir.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif"}} \
                   if sr_dir.exists() else set()

        overlap = test_names & sr_names
        if not overlap:
            logger.info(f"  [SKIP YOLO] {exp_name} — sin imágenes del test set en {sr_dir}")
            continue

        # Crear carpeta temporal con solo las imágenes del test set
        tmp_sr_test = ROOT / f"results/sr/{exp_name}_test_only"
        tmp_sr_test.mkdir(parents=True, exist_ok=True)
        for p in sr_dir.iterdir():
            if p.stem in overlap:
                dst = tmp_sr_test / p.name
                if not dst.exists():
                    import shutil
                    shutil.copy2(p, dst)

        _eval(tmp_sr_test, f"{exp_name}_yolo")


def phase_prepare_test_lr(logger: logging.Logger) -> dict[int, Path]:
    """
    Genera versiones LR (x2 y x4) de las imágenes del test set usando el
    mismo pipeline de degradación que se usó para el dataset principal.
    Devuelve {scale: directorio_lr_test}.
    """
    import cv2
    import numpy as np

    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    test_imgs = sorted(
        p for p in TEST_IMG.iterdir() if p.suffix.lower() in SUPPORTED_EXTS
    )
    if not test_imgs:
        logger.warning("No hay imágenes en el test set para degradar.")
        return {}

    lr_dirs: dict[int, Path] = {}
    for scale in [2, 4]:
        lr_test_dir = ROOT / f"data/lr_test/x{scale}"
        lr_test_dir.mkdir(parents=True, exist_ok=True)
        lr_dirs[scale] = lr_test_dir

        existing = {p.stem for p in lr_test_dir.iterdir()} if lr_test_dir.exists() else set()
        pending  = [p for p in test_imgs if p.stem not in existing]

        if not pending:
            logger.info(f"  [SKIP] LR test x{scale} ya existen en {lr_test_dir}")
            continue

        logger.info(f"  Generando LR x{scale} del test set ({len(pending)} imágenes) ...")
        for img_path in pending:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.warning(f"    No se pudo leer: {img_path.name}")
                continue
            h, w = img.shape[:2]
            # Blur + ruido (consistente con degrade.py: blur_noise)
            blur_k = 3 + (scale - 2)   # 3 para x2, 5 para x4
            if blur_k % 2 == 0:
                blur_k += 1
            img_blur = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
            noise    = np.random.normal(0, 5.0, img_blur.shape).astype(np.float32)
            img_noisy= np.clip(img_blur.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            lr       = cv2.resize(img_noisy, (w // scale, h // scale),
                                  interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(str(lr_test_dir / img_path.name), lr)

        logger.info(f"  LR test x{scale} guardadas en: {lr_test_dir}")

    return lr_dirs


def phase_sr_test(lr_dirs: dict[int, Path], ckpt_map: dict[int, Path],
                  logger: logging.Logger) -> dict[str, Path]:
    """
    Aplica SR sobre las versiones LR del test set para poder evaluar YOLO.
    Genera carpetas results/sr/<model>_test_x<scale>/ con las imágenes SR.
    Devuelve {exp_name_test: directorio_sr}.
    """
    sr_test_dirs: dict[str, Path] = {}

    models_to_run = [
        # (nombre, modelo, escala, checkpoint_propio)
        ("stablesr_x2_test",              "stablesr",  2, None),
        ("stablesr_x4_test",              "stablesr",  4, None),
        ("resshift_pretrained_x2_test",   "resshift",  2, None),
        ("resshift_pretrained_x4_test",   "resshift",  4, None),
        ("resshift_finetuned_x2_test",    "resshift",  2, ckpt_map.get(2)),
        ("resshift_finetuned_x4_test",    "resshift",  4, ckpt_map.get(4)),
    ]

    for exp_name, model, scale, ckpt in models_to_run:
        lr_dir = lr_dirs.get(scale)
        if lr_dir is None or not lr_dir.exists():
            logger.warning(f"  [SKIP SR test] {exp_name} — no hay LR test x{scale}")
            continue

        # Si es finetuned y no hay checkpoint, saltar
        if "finetuned" in exp_name and not ckpt:
            logger.info(f"  [SKIP SR test] {exp_name} — sin checkpoint finetuneado")
            continue

        out_dir = SR_DIR / exp_name
        if out_dir.exists() and any(out_dir.iterdir()):
            logger.info(f"  [SKIP SR test] {exp_name} — ya existe")
            sr_test_dirs[exp_name] = out_dir
            continue

        if model == "stablesr":
            cmd = [
                *PYTHON_STABLESR, str(SCRIPTS_SR / "run_stablesr.py"),
                "--input_dir",  str(lr_dir),
                "--output_dir", str(SR_DIR),
                "--checkpoint", str(STABLESR_CKPT),
                "--vqgan_ckpt", str(VQGAN_CKPT),
                "--scale",      str(scale),
                "--exp_name",   exp_name,
            ]
        else:  # resshift
            cmd = [
                *PYTHON_RESSHIFT, str(SCRIPTS_SR / "run_resshift.py"),
                "--input_dir",  str(lr_dir),
                "--output_dir", str(SR_DIR),
                "--scale",      str(scale),
                "--exp_name",   exp_name,
            ]
            if ckpt:
                cmd += ["--checkpoint", str(ckpt)]

        ok = run(cmd, label=f"SR test — {exp_name}", logger=logger)
        if ok:
            sr_test_dirs[exp_name] = out_dir

    return sr_test_dirs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pipeline completo de superresolución SR → métricas → YOLO"
    )
    p.add_argument(
        "--phases", nargs="+",
        choices=["stablesr", "resshift_pretrained", "train", "resshift_finetuned",
                 "metrics", "yolo", "compare"],
        default=None,
        help="Fases a ejecutar (por defecto: todas). "
             "Ejemplo: --phases stablesr metrics yolo"
    )
    p.add_argument("--skip_sr",    action="store_true",
                   help="Omitir todas las fases de inferencia SR")
    p.add_argument("--skip_train", action="store_true",
                   help="Omitir el finetune de ResShift")
    p.add_argument("--iterations", type=int, default=5000,
                   help="Iteraciones de finetune ResShift (default: 5000)")
    p.add_argument("--env_stablesr", type=str, default=None,
                   help="Nombre del entorno conda para StableSR (ej. stablesr)")
    p.add_argument("--env_resshift", type=str, default=None,
                   help="Nombre del entorno conda para ResShift (ej. resshift)")
    p.add_argument("--log_level",  default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_file = ROOT / "results/logs/pipeline.log"
    logger = setup_logging(log_file, args.log_level)

    logger.info("=" * 60)
    logger.info("PIPELINE SR  —  inicio")
    logger.info("=" * 60)

    global PYTHON_STABLESR, PYTHON_RESSHIFT
    if args.env_stablesr:
        PYTHON_STABLESR = ["conda", "run", "--no-capture-output", "-n", args.env_stablesr, "python"]
        logger.info(f"Usando conda env para StableSR: {args.env_stablesr}")
    if args.env_resshift:
        PYTHON_RESSHIFT = ["conda", "run", "--no-capture-output", "-n", args.env_resshift, "python"]
        logger.info(f"Usando conda env para ResShift: {args.env_resshift}")

    # Determinar qué fases correr
    all_phases = ["stablesr", "resshift_pretrained", "train",
                  "resshift_finetuned", "metrics", "yolo", "compare"]
    phases = set(args.phases) if args.phases else set(all_phases)

    if args.skip_sr:
        phases -= {"stablesr", "resshift_pretrained", "resshift_finetuned"}
    if args.skip_train:
        phases -= {"train", "resshift_finetuned"}

    logger.info(f"Fases activas: {sorted(phases)}")

    done_sr: set[str] = set()
    ckpt_map: dict[int, Path] = {}
    lr_test_dirs: dict[int, Path] = {}

    # ── 1. StableSR ─────────────────────────────────────────────────────────────
    if "stablesr" in phases:
        logger.info("\n" + "=" * 60)
        logger.info("FASE 1 — StableSR (inferencia)")
        logger.info("=" * 60)
        done_sr.update(phase_stablesr(SR_EXPERIMENTS, logger))

    # ── 2. ResShift pretrained ──────────────────────────────────────────────
    if "resshift_pretrained" in phases:
        logger.info("\n" + "=" * 60)
        logger.info("FASE 2 — ResShift (pretrained, descarga automática)")
        logger.info("=" * 60)
        done_sr.update(phase_resshift_pretrained(SR_EXPERIMENTS, logger))

    # ── 3. Finetune ResShift ──────────────────────────────────────────────
    if "train" in phases:
        logger.info("\n" + "=" * 60)
        logger.info("FASE 3 — ResShift (finetune sobre dataset propio)")
        logger.info("=" * 60)
        ckpt_map = phase_resshift_finetune(
            scales=[2, 4],
            iterations=args.iterations,
            logger=logger,
        )

    # ── 4. ResShift finetuned — inferencia ────────────────────────────────
    if "resshift_finetuned" in phases and ckpt_map:
        logger.info("\n" + "=" * 60)
        logger.info("FASE 4 — ResShift finetuned (inferencia)")
        logger.info("=" * 60)
        done_sr.update(phase_resshift_finetuned(SR_EXPERIMENTS, ckpt_map, logger))

    # ── 5. Métricas ───────────────────────────────────────────────────
    if "metrics" in phases:
        logger.info("\n" + "=" * 60)
        logger.info("FASE 5 — Cálculo de métricas PSNR / SSIM / LPIPS")
        logger.info("=" * 60)
        phase_metrics(SR_EXPERIMENTS, done_sr, logger)

    # ── 6. Preparar LR del test + SR test + YOLO ─────────────────────────
    if "yolo" in phases:
        logger.info("\n" + "=" * 60)
        logger.info("FASE 6a — Preparar LR del test set (x2 y x4)")
        logger.info("=" * 60)
        lr_test_dirs = phase_prepare_test_lr(logger)

        logger.info("\n" + "=" * 60)
        logger.info("FASE 6b — SR sobre las imágenes LR del test set")
        logger.info("=" * 60)
        sr_test_dirs = phase_sr_test(lr_test_dirs, ckpt_map, logger)

        logger.info("\n" + "=" * 60)
        logger.info("FASE 6c — Evaluación YOLO downstream")
        logger.info("=" * 60)
        # Baseline: test originales
        phase_yolo({"baseline_no_sr"}, logger)
        # SR test: cada carpeta generada
        for sr_exp, sr_dir in sr_test_dirs.items():
            phase_yolo({sr_exp}, logger)

    # ── 7. Tabla comparativa ──────────────────────────────────────────
    if "compare" in phases:
        logger.info("\n" + "=" * 60)
        logger.info("FASE 7 — Tabla comparativa final")
        logger.info("=" * 60)
        run(
            cmd=[sys.executable, str(SCRIPTS_SR / "compare_results.py")],
            label="compare_results",
            logger=logger,
        )

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SR  —  completado")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
