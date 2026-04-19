"""
05_run_sr.py — Inferencia SR con modelo StableSR finetuneado.

Carga el checkpoint entrenado, procesa cada imagen LR y guarda el resultado SR.

Uso:
    python scripts/05_run_sr.py \
        --lr_dir data/lr/x2 \
        --output_dir data/sr/x2 \
        --checkpoint models/stablesr_finetuned/exp0_default/checkpoints/best.ckpt \
        --scale 2 \
        --sampler_steps 20 \
        --exp_name exp0_default
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

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


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def check_cuda() -> "torch.device":
    import torch
    if torch.cuda.is_available():
        logging.info(f"CUDA disponible: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    logging.warning("CUDA NO disponible.")
    return torch.device("cpu")


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, mixed_precision: str, device: "torch.device"):
    """
    Carga el UNet finetuneado desde el checkpoint guardado por 04_train_stablesr.py
    y lo combina con los componentes base de Stable Diffusion.
    """
    import torch
    from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler

    logger = logging.getLogger(__name__)
    logger.info(f"Cargando checkpoint: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    dtype = torch.float16 if mixed_precision == "fp16" else torch.float32

    # Cargamos el pipe base
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=dtype,
        safety_checker=None,
    )

    # Inyectamos los pesos finetuneados en el UNet
    if "unet" in ckpt:
        pipe.unet.load_state_dict(ckpt["unet"], strict=False)
        epoch_info = ckpt.get("epoch", "desconocida")
        loss_info = ckpt.get("loss", "desconocida")
        logger.info(f"  Pesos UNet cargados (epoch={epoch_info}, loss={loss_info:.6f})")
    else:
        logger.warning("El checkpoint no contiene 'unet'. Se usa el UNet base sin finetuning.")

    pipe = pipe.to(device)
    pipe.unet.eval()
    pipe.vae.eval()

    return pipe


# ---------------------------------------------------------------------------
# Inferencia SR por imagen
# ---------------------------------------------------------------------------

def run_inference_single(
    pipe,
    lr_path: Path,
    scale: int,
    sampler_steps: int,
    device: "torch.device",
    mixed_precision: str,
) -> np.ndarray:
    """
    Devuelve la imagen SR como array numpy (H, W, C) uint8.
    """
    import torch
    from PIL import Image
    from torchvision import transforms

    dtype = torch.float16 if mixed_precision == "fp16" else torch.float32

    lr_img = Image.open(lr_path).convert("RGB")
    # Upscale LR al tamaño HR esperado con interpolación bicúbica como input condicionante
    hr_size = (lr_img.width * scale, lr_img.height * scale)
    lr_upscaled = lr_img.resize(hr_size, Image.BICUBIC)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    lr_tensor = to_tensor(lr_upscaled).unsqueeze(0).to(device, dtype=dtype)

    pipe.scheduler.set_timesteps(sampler_steps)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(mixed_precision == "fp16")):
        # Encode → latents de la imagen LR upscaled
        latents = pipe.vae.encode(lr_tensor).latent_dist.sample() * 0.18215
        # Denoising loop
        for t in pipe.scheduler.timesteps:
            noise_pred = pipe.unet(latents, t, encoder_hidden_states=None).sample
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        # Decode
        decoded = pipe.vae.decode(latents / 0.18215).sample

    # De [-1,1] → [0,255]
    decoded = (decoded.float().clamp(-1, 1) + 1) / 2
    decoded = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
    decoded = (decoded * 255).astype(np.uint8)
    return decoded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia SR con StableSR finetuneado.")
    parser.add_argument("--lr_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--sampler_steps", type=int, default=20)
    parser.add_argument("--mixed_precision", default="fp16", choices=["fp16", "no"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--exp_name", default="sr_inference")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    log_file = Path("results/logs") / f"{args.exp_name}.log"
    setup_logging(log_file, args.log_level)
    fix_seed(args.seed)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("SCRIPT 05 — INFERENCIA SR")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    import torch
    device = check_cuda() if args.device == "cuda" else torch.device("cpu")

    out_dir = args.output_dir / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        logger.error(f"Checkpoint no encontrado: {args.checkpoint}")
        sys.exit(1)

    pipe = load_model(args.checkpoint, args.mixed_precision, device)

    lr_paths = sorted(p for p in args.lr_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not lr_paths:
        logger.error(f"No se encontraron imágenes LR en {args.lr_dir}.")
        sys.exit(1)

    logger.info(f"Imágenes LR encontradas: {len(lr_paths)}")
    processed, skipped = 0, 0

    from PIL import Image
    import cv2

    for lr_path in tqdm(lr_paths, desc="Inferencia SR", unit="img"):
        try:
            sr_array = run_inference_single(
                pipe, lr_path, args.scale, args.sampler_steps, device, args.mixed_precision
            )
            # Guardar con nombre original (sin el sufijo _xN)
            stem = lr_path.stem
            if stem.endswith(f"_x{args.scale}"):
                stem = stem[: -len(f"_x{args.scale}")]
            out_path = out_dir / f"{stem}.png"
            # Convertir de RGB a BGR para OpenCV
            cv2.imwrite(str(out_path), cv2.cvtColor(sr_array, cv2.COLOR_RGB2BGR))
            processed += 1
            logger.debug(f"  Guardado: {out_path}")
        except Exception as e:
            logger.warning(f"Error procesando {lr_path.name}: {e}. Se omite.")
            skipped += 1

    logger.info(f"Inferencia completada: {processed} imágenes SR, {skipped} omitidas.")
    logger.info(f"Resultados en: {out_dir}")


if __name__ == "__main__":
    main()
