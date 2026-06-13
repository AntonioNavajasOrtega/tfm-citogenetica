"""
run_resshift.py — Inferencia de super-resolución con ResShift.
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
    p = argparse.ArgumentParser(description="Inferencia con ResShift")
    p.add_argument("--input_dir", type=Path, required=True, help="Directorio de imágenes LR de entrada")
    p.add_argument("--output_dir", type=Path, required=True, help="Directorio de salida para imágenes SR")
    p.add_argument("--checkpoint", type=str, required=True, help="Ruta al checkpoint de ResShift (modelo preentrenado o finetuneado)")
    
    # Parámetros de inferencia para ResShift
    p.add_argument("--task", type=str, default="srx2", help="Tarea a resolver (ej. srx2, srx4)")
    p.add_argument("--steps", type=int, default=15, help="Número de pasos de difusión para ResShift")
    p.add_argument("--chop", action="store_true", help="Habilitar 'chop' (parcheado) para procesar imágenes grandes con poca VRAM")
    
    p.add_argument("--scale", type=int, default=2, help="Factor de escala")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exp_name", type=str, default="resshift_eval")
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

    out_dir = args.output_dir / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.input_dir.exists():
        logger.error(f"El directorio de entrada no existe: {args.input_dir}")
        sys.exit(1)

    # --- Integración con ResShift ---
    resshift_dir = Path("c:/TFM/tfm-citogenetica/ResShift").resolve()
    if str(resshift_dir) not in sys.path:
        sys.path.insert(0, str(resshift_dir))
    
    import os
    from omegaconf import OmegaConf
    from sampler import ResShiftSampler

    logger.info(f"Cargando modelo ResShift desde {args.checkpoint} para tarea {args.task}")
    
    # Seleccionar configuración base según la tarea
    if "bic" in args.task.lower():
        config_path = resshift_dir / "configs/bicx4_swinunet_lpips.yaml"
    else:
        config_path = resshift_dir / "configs/realsr_swinunet_realesrgan256.yaml"
        
    configs = OmegaConf.load(str(config_path))
    configs.model.ckpt_path = str(Path(args.checkpoint).resolve())
    configs.diffusion.params.sf = args.scale
    configs.autoencoder.ckpt_path = str(resshift_dir / "weights/autoencoder_vq_f4.pth")
    
    chop_size = 512 * (4 // args.scale) if args.chop else -1
    chop_stride = (512 - 64) * (4 // args.scale) if args.chop else -1

    # Cambiar de directorio temporalmente por si ResShift depende de paths relativos
    original_cwd = os.getcwd()
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
                padding_offset=configs.model.params.get('lq_size', 64),
        )
        
        logger.info("Iniciando inferencia (el progreso lo maneja ResShift)...")
        model.inference(
                in_path=str(args.input_dir.resolve()),
                out_path=str(out_dir.resolve()),
                bs=1,
                noise_repeat=False
        )
    finally:
        os.chdir(original_cwd)

    logger.info(f"Inferencia completada. Resultados en: {out_dir}")

if __name__ == "__main__":
    main()
