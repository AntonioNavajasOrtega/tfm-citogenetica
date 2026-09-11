"""
run_resshift.py — Inferencia de super-resolución con ResShift.

Delega en inference_resshift.py del repositorio ResShift (que auto-descarga los
pesos si no existen). Soporta x2 y x4, tarea 'realsr' por defecto.

Uso — reconstrucción (LR → HR):
    python scripts/diffusion/run_resshift.py \\
        --input_dir  data/lr/x2 \\
        --output_dir results/sr \\
        --scale 2 \\
        --version v3 \\
        --exp_name resshift_pretrained_x2_recon

Uso — con checkpoint finetuneado propio:
    python scripts/diffusion/run_resshift.py \\
        --input_dir  data/lr/x2 \\
        --output_dir results/sr \\
        --scale 2 \\
        --checkpoint models/resshift_finetuned/resshift_train_x2/ckpts/best.pth \\
        --exp_name resshift_finetuned_x2_recon

Uso — upscale de originales HR:
    python scripts/diffusion/run_resshift.py \\
        --input_dir  data/raw/unmarked \\
        --output_dir results/sr \\
        --scale 4 \\
        --exp_name resshift_x4_upscale
"""

import argparse
import logging
import os
from pyexpat import model
import sys
from pathlib import Path

import torch

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Versiones disponibles en ResShift (todas x4 salvo indicación)
_VERSION_STEPS = {"v1": 15, "v2": 15, "v3": 4}


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


def fix_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inferencia con ResShift")
    p.add_argument("--input_dir",  type=Path, required=True,
                   help="Directorio de imágenes LR de entrada (o HR para upscale)")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Directorio raíz de salida para imágenes SR")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Ruta a un checkpoint propio (.pth). Si no se da, se usa "
                        "el checkpoint oficial de ResShift (se descarga automáticamente).")

    # Parámetros de inferencia para ResShift
    p.add_argument("--scale",   type=int, default=4, choices=[2, 4],
                   help="Factor de escala SR (2 o 4; por defecto 4)")
    p.add_argument("--version", type=str, default="v3",
                   choices=["v1", "v2", "v3"],
                   help="Versión del checkpoint preentrenado oficial: v1/v2 (15 pasos), "
                        "v3 (4 pasos, más rápido). Ignorado si se pasa --checkpoint.")
    p.add_argument("--steps",   type=int, default=None,
                   help="Pasos de difusión. Por defecto usa los pasos de la versión elegida.")
    p.add_argument("--chop_size", type=int, default=512, choices=[512, 256, 64],
                   help="Tamaño de tile para imágenes grandes (chop). Por defecto 512.")

    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--exp_name",  type=str, default="resshift_eval")
    p.add_argument("--log_level", default="INFO")

    return p.parse_args()


def main():
    args = parse_args()
    log_file = Path("results/logs") / f"{args.exp_name}.log"
    logger = setup_logging(log_file, args.log_level)
    fix_seed(args.seed)

    logger.info("=" * 60)
    logger.info("INFERENCIA — ResShift")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    # Directorio de salida con nombre del experimento
    out_dir = args.output_dir / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_dir.exists():
        logger.error(f"El directorio de entrada no existe: {args.input_dir}")
        sys.exit(1)

    img_paths = [p for p in args.input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]
    if not img_paths:
        logger.error(f"No hay imágenes en {args.input_dir}")
        sys.exit(1)
    logger.info(f"Imágenes a procesar: {len(img_paths)}")

    # --- Integración con ResShift vía importación directa ---
    resshift_dir = Path("c:/TFM/ResShift").resolve()
    if not resshift_dir.exists():
        logger.error(f"No se encuentra el directorio de ResShift: {resshift_dir}")
        sys.exit(1)

    if str(resshift_dir) not in sys.path:
        sys.path.insert(0, str(resshift_dir))

    from omegaconf import OmegaConf
    from sampler import ResShiftSampler

    # ── Selección de configuración y checkpoint ─────────────────────────────
    if args.scale == 2:
        # x2: config específico de x2 (lq_size=128, sf=2, steps=4)
        config_path = resshift_dir / "configs/realsr_realesrgan256_x2.yaml"
        ckpt_name   = "resshift_realsrx2_s4_v1.pth"
        ckpt_url    = ("https://github.com/zsyOAOA/ResShift/releases/download/"
                       "v2.0/resshift_realsrx2_s4_v1.pth")
        steps_default = 4
    else:
        # x4: config estándar
        if args.version == "v3":
            config_path = resshift_dir / "configs/realsr_swinunet_realesrgan256_journal.yaml"
        else:
            config_path = resshift_dir / "configs/realsr_swinunet_realesrgan256.yaml"
        ckpt_name   = f"resshift_realsrx4_s{_VERSION_STEPS[args.version]}_{args.version}.pth"
        ckpt_url    = ("https://github.com/zsyOAOA/ResShift/releases/download/v2.0/"
                       f"resshift_realsrx4_s{_VERSION_STEPS[args.version]}_{args.version}.pth")
        steps_default = _VERSION_STEPS[args.version]

    if not config_path.exists():
        logger.error(f"Config de ResShift no encontrado: {config_path}")
        sys.exit(1)

    configs = OmegaConf.load(str(config_path))

    # ── Checkpoint ──────────────────────────────────────────────────────────
    if args.checkpoint:
        # Checkpoint propio (finetuneado)
        ckpt_path = Path(args.checkpoint).resolve()
        if not ckpt_path.exists():
            logger.error(f"Checkpoint propio no encontrado: {ckpt_path}")
            sys.exit(1)
        logger.info(f"Usando checkpoint propio: {ckpt_path}")
    else:
        # Checkpoint oficial: descargar si no existe
        from basicsr.utils.download_util import load_file_from_url
        weights_dir = resshift_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        ckpt_path = weights_dir / ckpt_name
        if not ckpt_path.exists():
            logger.info(f"Descargando checkpoint oficial: {ckpt_name} ...")
            load_file_from_url(url=ckpt_url, model_dir=weights_dir,
                               progress=True, file_name=ckpt_name)
        else:
            logger.info(f"Checkpoint oficial ya presente: {ckpt_path}")

    # ── VQ-GAN autoencoder ───────────────────────────────────────────────────
    vqgan_url  = ("https://github.com/zsyOAOA/ResShift/releases/download/"
                  "v2.0/autoencoder_vq_f4.pth")
    vqgan_path = resshift_dir / "weights" / "autoencoder_vq_f4.pth"
    if not vqgan_path.exists():
        from basicsr.utils.download_util import load_file_from_url
        logger.info("Descargando autoencoder VQ-GAN ...")
        load_file_from_url(url=vqgan_url, model_dir=resshift_dir / "weights",
                           progress=True, file_name="autoencoder_vq_f4.pth")

    # ── Configurar OmegaConf ─────────────────────────────────────────────────
    configs.model.ckpt_path        = str(ckpt_path)
    configs.autoencoder.ckpt_path  = str(vqgan_path)
    configs.diffusion.params.sf    = args.scale

    n_steps = args.steps if args.steps is not None else steps_default
    configs.diffusion.params.steps = n_steps

    # ── Chop (tiling) ────────────────────────────────────────────────────────
    chop_size   = args.chop_size * (4 // args.scale)
    chop_stride = int(chop_size * (1 - 64 / args.chop_size))
    logger.info(f"Chop size/stride: {chop_size}/{chop_stride}")

    # ── Inferencia ───────────────────────────────────────────────────────────
    original_cwd = os.getcwd()
    in_path_resolved  = args.input_dir.resolve()
    out_path_resolved = out_dir.resolve()
    os.chdir(str(resshift_dir))

    try:
        model = ResShiftSampler(
            configs,
            sf=args.scale,
            chop_size=chop_size,
            chop_stride=chop_stride,
            chop_bs=1,
            use_amp=True,
            seed=args.seed,
            padding_offset=configs.model.params.get("lq_size", 64),
        )
        model.model.half() 
        model.autoencoder.half()

        logger.info("Iniciando inferencia ResShift...")
        print("AMP dtype:", torch.get_default_dtype())
        print("Device:", next(model.model.parameters()).device)
        model.inference(
        in_path=in_path_resolved,
        out_path=out_path_resolved,
        bs=1,
        noise_repeat=False,
)
    finally:
        os.chdir(original_cwd)

    logger.info(f"Inferencia completada. Resultados en: {out_dir}")


if __name__ == "__main__":
    main()
