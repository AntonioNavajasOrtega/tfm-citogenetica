"""
run_stablesr.py — Inferencia de super-resolución con StableSR.

Soporta argumentos clave de StableSR como ddpm_steps, sampler, dec_w y colorfix.
"""

import argparse
import logging
import sys
from pathlib import Path
from tqdm import tqdm

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

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
    p = argparse.ArgumentParser(description="Inferencia con StableSR")
    p.add_argument("--input_dir", type=Path, required=True, help="Directorio de imágenes LR de entrada")
    p.add_argument("--output_dir", type=Path, required=True, help="Directorio de salida para imágenes SR")
    p.add_argument("--checkpoint", type=str, default=None, help="Ruta al checkpoint de StableSR (opcional, si no usa el preentrenado)")
    
    # Parámetros específicos de StableSR requeridos
    p.add_argument("--ddpm_steps", type=int, default=200, help="Número de pasos de inferencia (ddpm steps)")
    p.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim", "spaced"], help="Scheduler o sampler a usar")
    p.add_argument("--dec_w", type=float, default=0.5, help="Weight para la fusión de decodificación (dec_w)")
    p.add_argument("--colorfix", type=str, default="wavelet", choices=["wavelet", "adain", "none"], help="Método de corrección de color (colorfix)")
    
    p.add_argument("--scale", type=int, default=2, help="Factor de escala")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exp_name", type=str, default="stablesr_eval")
    p.add_argument("--log_level", default="INFO")
    
    return p.parse_args()

def main():
    args = parse_args()
    log_file = Path("results/logs") / f"{args.exp_name}.log"
    logger = setup_logging(log_file, args.log_level)
    fix_seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("INFERENCIA — StableSR")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    out_dir = args.output_dir / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.input_dir.exists():
        logger.error(f"El directorio de entrada no existe: {args.input_dir}")
        sys.exit(1)
        
    img_paths = sorted([p for p in args.input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
    if not img_paths:
        logger.error(f"No se encontraron imágenes en {args.input_dir}")
        sys.exit(1)
        
    logger.info(f"Imágenes a procesar: {len(img_paths)}")

    # --- Integración con StableSR vía subprocess ---
    import subprocess
    stablesr_dir = Path("c:/TFM/tfm-citogenetica/StableSR").resolve()
    
    if args.sampler.lower() == "ddim":
        script_name = "sr_val_ddim_text_T_negativeprompt.py"
    else:
        # Default a ddpm
        script_name = "sr_val_ddpm_text_T_vqganfin_old.py"
        
    script_path = stablesr_dir / "scripts" / script_name
    
    if not script_path.exists():
        logger.error(f"El script base de StableSR no existe: {script_path}")
        sys.exit(1)
        
    # Construir el comando a ejecutar
    cmd = [
        sys.executable, str(script_path),
        "--init-img", str(args.input_dir.resolve()),
        "--outdir", str(out_dir.resolve()),
        "--dec_w", str(args.dec_w),
        "--colorfix_type", args.colorfix,
        "--seed", str(args.seed)
    ]
    
    if args.sampler.lower() == "ddim":
        cmd.extend(["--ddim_steps", str(args.ddpm_steps)])
    
    if args.checkpoint:
        cmd.extend(["--ckpt", str(Path(args.checkpoint).resolve())])
        
    logger.info(f"Lanzando inferencia delegada a StableSR...")
    logger.debug(f"Comando: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, cwd=str(stablesr_dir), check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Fallo durante la ejecución de StableSR: {e}")
        sys.exit(1)

    logger.info(f"Inferencia completada. Resultados en: {out_dir}")

if __name__ == "__main__":
    main()
