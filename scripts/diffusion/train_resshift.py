"""
train_resshift.py — Script de entrenamiento/finetuning para ResShift.
"""

import argparse
import logging
import sys
from pathlib import Path

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
    p = argparse.ArgumentParser(description="Entrenamiento/Finetuning con ResShift")
    p.add_argument("--config", type=Path, required=True, help="Ruta al archivo de configuración YAML de ResShift")
    p.add_argument("--hr_dir", type=Path, default=Path("data/processed"), help="Directorio con imágenes HR originales")
    p.add_argument("--lr_dir", type=Path, default=Path("data/lr/x2"), help="Directorio con imágenes LR de entrenamiento")
    p.add_argument("--output_dir", type=Path, default=Path("models/resshift_finetuned"), help="Directorio principal de salida para checkpoints")
    
    # Hiperparámetros que pueden sobrescribir la configuración base
    p.add_argument("--batch_size", type=int, default=4, help="Tamaño de batch (sobrescribe YAML si se especifica)")
    p.add_argument("--epochs", type=int, default=100, help="Número de épocas")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate (tasa de aprendizaje)")
    p.add_argument("--resume_from", type=str, default=None, help="Ruta a checkpoint para reanudar el entrenamiento")
    
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exp_name", type=str, default="resshift_train")
    p.add_argument("--log_level", default="INFO")
    
    return p.parse_args()

def main():
    args = parse_args()
    log_file = Path("results/logs") / f"{args.exp_name}.log"
    logger = setup_logging(log_file, args.log_level)
    fix_seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO — ResShift")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    out_dir = args.output_dir / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.config.exists():
        logger.error(f"El archivo de configuración no existe: {args.config}")
        sys.exit(1)
        
    # --- Integración con ResShift ---
    resshift_dir = Path("c:/TFM/tfm-citogenetica/ResShift").resolve()
    if str(resshift_dir) not in sys.path:
        sys.path.insert(0, str(resshift_dir))
    
    import os
    from omegaconf import OmegaConf
    from utils.util_common import get_obj_from_str

    logger.info(f"Cargando configuración base desde: {args.config}")
    configs = OmegaConf.load(str(args.config.resolve()))
    
    # Sobrescribir configuración con los argumentos proporcionados
    configs.save_dir = str(out_dir.resolve())
    if args.resume_from:
        configs.resume = str(Path(args.resume_from).resolve())
    else:
        configs.resume = ""
        
    # Asignar los dataloaders para apuntar a los directorios del usuario si es necesario
    # (En la práctica de ResShift los directorios se ponen en el yaml, 
    # pero podemos intentar forzarlos si se usan en la config estándar)
    if hasattr(configs, 'data') and hasattr(configs.data, 'train'):
        if hasattr(configs.data.train, 'params'):
            configs.data.train.params.dir_hr = str(args.hr_dir.resolve())
            configs.data.train.params.dir_lr = str(args.lr_dir.resolve())
            configs.data.train.params.batch_size = args.batch_size
            
    # Sobrescribir hyperparams básicos
    if hasattr(configs, 'trainer') and hasattr(configs.trainer, 'epochs'):
        configs.trainer.epochs = args.epochs
    
    # Cambiar de directorio temporalmente por si ResShift depende de paths relativos
    original_cwd = os.getcwd()
    os.chdir(str(resshift_dir))
    
    try:
        logger.info("Inicializando modelo, optimizadores y trainer...")
        trainer = get_obj_from_str(configs.trainer.target)(configs)
        
        logger.info(f"Iniciando el bucle de entrenamiento por {args.epochs} épocas...")
        trainer.train()
    except Exception as e:
        logger.error(f"Error en el entrenamiento: {e}")
        raise e
    finally:
        os.chdir(original_cwd)

    logger.info(f"Entrenamiento completado. Checkpoints en: {out_dir}")

if __name__ == "__main__":
    main()
