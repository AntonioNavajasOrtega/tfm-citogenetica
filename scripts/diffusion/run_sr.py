"""
run_sr.py — super-resolución con stable-diffusion-x4-upscaler de huggingface.

toma las imágenes de test (512×512), las reduce internamente a 128×128
y aplica SR para obtener 512×512 de mayor calidad visual.

schedulers disponibles: ddim (determinista), pndm, lms, ddpm (estocástico)

uso:
    python scripts/run_sr.py --scheduler ddim --steps 50 --exp_name sr_ddim_50
    python scripts/run_sr.py --scheduler ddpm --steps 50 --exp_name sr_ddpm_50
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# tamaño de entrada al modelo (diseñado para 128→512)
MODEL_INPUT_SIZE = 128


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
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_pipeline(scheduler_name: str, checkpoint: str | None, mixed_precision: str):
    """
    carga el pipeline de SR y sustituye el scheduler según el argumento.
    si se pasa --checkpoint carga pesos finetuneados sobre el base.
    """
    import torch
    from diffusers import (
        StableDiffusionUpscalePipeline,
        DDIMScheduler,
        PNDMScheduler,
        DDPMScheduler,
        LMSDiscreteScheduler,
    )

    logger = logging.getLogger(__name__)
    dtype = torch.float16 if mixed_precision == "fp16" else torch.float32

    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    logger.info(f"cargando modelo: {model_id}")

    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    # cargar pesos finetuneados si se pasan
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            logger.error(f"checkpoint no encontrado: {ckpt_path}")
            sys.exit(1)
        logger.info(f"cargando checkpoint finetuneado: {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location="cpu")
        if "unet" in state:
            pipe.unet.load_state_dict(state["unet"], strict=False)
            logger.info(f"  pesos unet cargados (epoch={state.get('epoch','?')})")

    # sustituir scheduler
    scheduler_map = {
        "ddim": DDIMScheduler,
        "pndm": PNDMScheduler,
        "ddpm": DDPMScheduler,
        "lms":  LMSDiscreteScheduler,
    }
    if scheduler_name not in scheduler_map:
        logger.error(f"scheduler desconocido: {scheduler_name}. opciones: {list(scheduler_map)}")
        sys.exit(1)

    sched_cls = scheduler_map[scheduler_name]
    pipe.scheduler = sched_cls.from_config(pipe.scheduler.config)
    logger.info(f"scheduler: {scheduler_name} ({sched_cls.__name__})")

    # optimizaciones de vram para rtx 5060 8gb
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
        logger.info("xformers activo")
    except Exception:
        logger.info("xformers no disponible, usando attention slicing")

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    logger.info(f"dispositivo: {device}")
    return pipe, device


def run_sr_image(pipe, img_path: Path, steps: int, noise_level: int, seed: int) -> Image.Image:
    """
    aplica sr a una imagen.
    redimensiona a 128×128 como input del modelo (diseñado para 128→512).
    devuelve imagen PIL 512×512.
    """
    import torch

    img = Image.open(img_path).convert("RGB")

    # el modelo espera input de 128×128 → output 512×512
    lr_img = img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.LANCZOS)

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator.manual_seed(seed)

    result = pipe(
        prompt="",          # sin prompt de texto, solo condicionado en imagen
        image=lr_img,
        num_inference_steps=steps,
        guidance_scale=0,   # sin classifier-free guidance (no hay texto)
        noise_level=noise_level,
        generator=generator,
    )
    return result.images[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="sr con stable-diffusion-x4-upscaler")
    p.add_argument("--input_dir",      type=Path, default=Path("data/test/images"),
                   help="imágenes de entrada (test set)")
    p.add_argument("--output_dir",     type=Path, default=Path("data/sr"),
                   help="directorio raíz de salida")
    p.add_argument("--scheduler",      default="ddim",
                   choices=["ddim", "pndm", "ddpm", "lms"])
    p.add_argument("--steps",          type=int, default=50,
                   help="pasos de inferencia del scheduler")
    p.add_argument("--noise_level",    type=int, default=20,
                   help="nivel de ruido añadido a la condición (20 es el default del paper)")
    p.add_argument("--checkpoint",     type=str, default=None,
                   help="ruta a checkpoint finetuneado (opcional)")
    p.add_argument("--mixed_precision",default="fp16", choices=["fp16", "no"])
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--log_level",      default="INFO")
    p.add_argument("--exp_name",       default=None,
                   help="nombre del experimento (default: sr_{scheduler}_{steps})")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # nombre del experimento por defecto es descriptivo
    if args.exp_name is None:
        args.exp_name = f"sr_{args.scheduler}_{args.steps}"

    log_file = Path("results/logs") / f"{args.exp_name}.log"
    logger = setup_logging(log_file, args.log_level)
    fix_seed(args.seed)

    logger.info("=" * 60)
    logger.info("SUPER-RESOLUCIÓN — stable-diffusion-x4-upscaler")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)
    logger.info(f"  input_size_model: {MODEL_INPUT_SIZE}×{MODEL_INPUT_SIZE} → 512×512")

    pipe, device = load_pipeline(args.scheduler, args.checkpoint, args.mixed_precision)

    out_dir = args.output_dir / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not img_paths:
        logger.error(f"no hay imágenes en {args.input_dir}")
        sys.exit(1)

    logger.info(f"imágenes a procesar: {len(img_paths)}")
    ok, fail = 0, 0

    for img_path in tqdm(img_paths, desc=f"sr [{args.exp_name}]", unit="img"):
        try:
            sr_img = run_sr_image(pipe, img_path, args.steps, args.noise_level, args.seed)
            out_path = out_dir / f"{img_path.stem}.png"
            sr_img.save(str(out_path))
            ok += 1
            logger.debug(f"  guardado: {out_path} ({sr_img.size})")
        except Exception as e:
            logger.warning(f"error en {img_path.name}: {e}")
            fail += 1

    logger.info(f"listo: {ok} procesadas, {fail} fallidas")
    logger.info(f"resultados en: {out_dir}")


if __name__ == "__main__":
    main()
