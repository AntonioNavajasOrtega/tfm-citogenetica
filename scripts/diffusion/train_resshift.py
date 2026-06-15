"""
train_resshift.py — Finetuning de ResShift sobre el dataset de citogenética.

Genera automáticamente el YAML de configuración adaptado al dataset local
y lanza el entrenamiento del trainer de ResShift.

Uso — finetune x4 (defecto):
    python scripts/diffusion/train_resshift.py \\
        --hr_dir data/raw/unmarked \\
        --lr_dir data/lr/x4 \\
        --scale  4 \\
        --iterations 5000 \\
        --exp_name resshift_finetune_x4

Uso — finetune x2:
    python scripts/diffusion/train_resshift.py \\
        --hr_dir  data/raw/unmarked \\
        --lr_dir  data/lr/x2 \\
        --scale   2 \\
        --iterations 5000 \\
        --exp_name resshift_finetune_x2

Uso — reanudar desde checkpoint:
    python scripts/diffusion/train_resshift.py \\
        --hr_dir  data/raw/unmarked \\
        --lr_dir  data/lr/x2 \\
        --scale   2 \\
        --resume_from models/resshift_finetuned/resshift_finetune_x2/ckpts/last.pth \\
        --exp_name resshift_finetune_x2_continued
"""

import argparse
import logging
import os
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
    p = argparse.ArgumentParser(description="Finetuning/Entrenamiento con ResShift")

    # Datos
    p.add_argument("--hr_dir",   type=Path, default=Path("data/raw/unmarked"),
                   help="Directorio con imágenes HR originales")
    p.add_argument("--lr_dir",   type=Path, default=None,
                   help="Directorio con imágenes LR de entrenamiento. "
                        "Si no se indica, ResShift genera la degradación on-the-fly.")
    p.add_argument("--val_hr_dir", type=Path, default=None,
                   help="Directorio HR de validación (opcional; si no, se usa hr_dir).")
    p.add_argument("--val_lr_dir", type=Path, default=None,
                   help="Directorio LR de validación (opcional).")

    # Escala
    p.add_argument("--scale", type=int, default=4, choices=[2, 4],
                   help="Factor de escala SR a entrenar (2 o 4)")

    # Configuración base del modelo
    p.add_argument("--config", type=Path, default=None,
                   help="YAML de configuración de ResShift. Si no se da, se elige "
                        "automáticamente según --scale.")
    p.add_argument("--pretrained_ckpt", type=str, default=None,
                   help="Checkpoint preentrenado de ResShift desde el que hacer finetune. "
                        "Si no se da, se descarga el oficial.")

    # Hiperparámetros de entrenamiento
    p.add_argument("--iterations", type=int, default=5000,
                   help="Número de iteraciones de entrenamiento (default: 5000)")
    p.add_argument("--batch_size", type=int, default=2,
                   help="Batch size (microbatch, default: 2)")
    p.add_argument("--lr",         type=float, default=5e-5,
                   help="Learning rate (default: 5e-5)")
    p.add_argument("--save_freq",  type=int, default=1000,
                   help="Guardar checkpoint cada N iteraciones (default: 1000)")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Ruta a checkpoint para reanudar el entrenamiento")

    # Salida y misc
    p.add_argument("--output_dir", type=Path, default=Path("models/resshift_finetuned"),
                   help="Directorio raíz de salida para checkpoints")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--exp_name",   type=str, default="resshift_finetune")
    p.add_argument("--log_level",  default="INFO")

    return p.parse_args()


def build_finetune_config(args, resshift_dir: Path, out_dir: Path) -> Path:
    """
    Genera un YAML de configuración de ResShift adaptado al dataset local.
    Hereda de la config base y sobreescribe los paths y los hiperparámetros
    de entrenamiento para que sea viable con un dataset pequeño.
    """
    from omegaconf import OmegaConf

    logger = logging.getLogger(__name__)

    # ── Config base según escala ─────────────────────────────────────────────
    if args.config and args.config.exists():
        base_config_path = args.config
    elif args.scale == 2:
        base_config_path = resshift_dir / "configs/realsr_realesrgan256_x2.yaml"
    else:
        base_config_path = resshift_dir / "configs/realsr_swinunet_realesrgan256.yaml"

    logger.info(f"Config base: {base_config_path}")
    configs = OmegaConf.load(str(base_config_path))

    # ── Rutas de datos ───────────────────────────────────────────────────────
    hr_dir_abs = str(args.hr_dir.resolve())
    # dir_paths espera una lista de directorios con imágenes HR
    configs.data.train.params.dir_paths   = [hr_dir_abs]
    configs.data.train.params.txt_file_path = []   # desactivar txt_file_path
    configs.data.train.params.im_exts     = ["JPG", "jpg", "png", "PNG", "jpeg"]

    # Validación
    val_hr = args.val_hr_dir or args.hr_dir
    val_lr = args.val_lr_dir
    configs.data.val.params.dir_path      = str(val_hr.resolve())
    configs.data.val.params.im_exts       = "jpg"
    if val_lr is not None:
        configs.data.val.params.extra_dir_path = str(val_lr.resolve())
    else:
        # ResShift puede hacer downscale on-the-fly; quitar extra_dir_path
        if hasattr(configs.data.val.params, "extra_dir_path"):
            del configs.data.val.params["extra_dir_path"]

    # ── Checkpoint inicial (preentrenado) ────────────────────────────────────
    if args.pretrained_ckpt:
        configs.model.ckpt_path = str(Path(args.pretrained_ckpt).resolve())
    # (si no hay pretrained_ckpt, queda como ~ y ResShift entrena desde cero)

    # ── Autoencoder VQ-GAN ───────────────────────────────────────────────────
    vqgan_path = resshift_dir / "weights" / "autoencoder_vq_f4.pth"
    configs.autoencoder.ckpt_path = str(vqgan_path)
    configs.autoencoder.tune_decoder = False
    configs.autoencoder.params.lora_tune_decoder = False

    # ── Hiperparámetros de entrenamiento ─────────────────────────────────────
    configs.train.iterations  = args.iterations
    configs.train.microbatch  = args.batch_size
    # batch [global_bs, num_gpus] — para 1 GPU usamos microbatch como global
    configs.train.batch       = [args.batch_size, 1]
    configs.train.lr          = args.lr
    configs.train.lr_min      = args.lr * 0.4
    configs.train.save_freq   = args.save_freq
    configs.train.val_freq = args.iterations + 1
    configs.train.num_workers = 2
    configs.train.log_freq    = [max(50, args.save_freq // 10),
                                  max(200, args.save_freq // 5),
                                  1]
    configs.train.use_amp     = True
    configs.train.seed        = args.seed
    configs.train.tf_logging  = False

    # ── Rutas de salida ──────────────────────────────────────────────────────
    configs.save_dir = str(out_dir.resolve())
    configs.resume   = str(Path(args.resume_from).resolve()) if args.resume_from else ""

    # Escribir YAML generado
    yaml_out = out_dir / "finetune_config.yaml"
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(configs, str(yaml_out))
    logger.info(f"YAML de finetune generado en: {yaml_out}")

    return yaml_out


def main():
    args = parse_args()
    log_file = Path("results/logs") / f"{args.exp_name}.log"
    logger = setup_logging(log_file, args.log_level)
    fix_seed(args.seed)

    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO/FINETUNE — ResShift")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    # Directorio de salida del experimento
    out_dir = args.output_dir / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validaciones de entrada
    if not args.hr_dir.exists():
        logger.error(f"Directorio HR no encontrado: {args.hr_dir}")
        sys.exit(1)
    if args.lr_dir and not args.lr_dir.exists():
        logger.error(f"Directorio LR no encontrado: {args.lr_dir}")
        sys.exit(1)

    # ── Configurar ResShift ──────────────────────────────────────────────────
    resshift_dir = Path("c:/TFM/ResShift").resolve()
    if not resshift_dir.exists():
        logger.error(f"No se encuentra el directorio de ResShift: {resshift_dir}")
        sys.exit(1)

    if str(resshift_dir) not in sys.path:
        sys.path.insert(0, str(resshift_dir))

    # Auto-descarga del VQ-GAN si no existe
    vqgan_path = resshift_dir / "weights" / "autoencoder_vq_f4.pth"
    if not vqgan_path.exists():
        from basicsr.utils.download_util import load_file_from_url
        (resshift_dir / "weights").mkdir(exist_ok=True)
        logger.info("Descargando autoencoder VQ-GAN ...")
        load_file_from_url(
            url="https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth",
            model_dir=resshift_dir / "weights",
            progress=True,
            file_name="autoencoder_vq_f4.pth",
        )

    # Generar YAML de finetune
    yaml_path = build_finetune_config(args, resshift_dir, out_dir)

    # ── Lanzar entrenamiento ─────────────────────────────────────────────────
    from omegaconf import OmegaConf
    from utils.util_common import get_obj_from_str

    logger.info(f"Cargando configuración desde: {yaml_path}")
    configs = OmegaConf.load(str(yaml_path))

    original_cwd = os.getcwd()
    os.chdir(str(resshift_dir))

    try:
        logger.info("Inicializando trainer...")
        trainer = get_obj_from_str(configs.trainer.target)(configs)
        logger.info(f"Iniciando entrenamiento por {args.iterations} iteraciones...")
        trainer.train()
    except Exception as e:
        logger.error(f"Error en el entrenamiento: {e}")
        raise
    finally:
        os.chdir(original_cwd)

    logger.info(f"Entrenamiento completado. Checkpoints en: {out_dir}")


if __name__ == "__main__":
    main()
