"""
run_sr.py — inferencia sr con el modelo stablesr finetuneado.

carga el checkpoint entrenado, procesa cada imagen lr y guarda el resultado sr.

uso:
    python scripts/run_sr.py \
        --lr_dir data/lr/x2 \
        --output_dir data/sr/x2 \
        --checkpoint models/stablesr_finetuned/exp0_default/checkpoints/best.ckpt \
        --scale 2 \
        --exp_name exp0_default
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


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
        logging.info(f"cuda: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    logging.warning("sin cuda")
    return torch.device("cpu")


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_model(checkpoint_path: Path, mixed_precision: str, device: "torch.device"):
    """
    carga el unet finetuneado e inyecta pesos en el pipeline base de sd.
    """
    import torch
    from diffusers import StableDiffusionPipeline

    logger = logging.getLogger(__name__)
    logger.info(f"cargando checkpoint: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    dtype = torch.float16 if mixed_precision == "fp16" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=dtype,
        safety_checker=None,
    )

    # inyecta los pesos del unet finetuneado
    if "unet" in ckpt:
        pipe.unet.load_state_dict(ckpt["unet"], strict=False)
        epoch_info = ckpt.get("epoch", "desconocida")
        loss_info  = ckpt.get("loss", float("nan"))
        logger.info(f"  pesos cargados (epoch={epoch_info}, loss={loss_info:.6f})")
    else:
        logger.warning("sin clave 'unet' en checkpoint, se usa el unet base")

    pipe = pipe.to(device)
    pipe.unet.eval()
    pipe.vae.eval()
    return pipe


def run_inference_single(
    pipe,
    lr_path: Path,
    scale: int,
    sampler_steps: int,
    device: "torch.device",
    mixed_precision: str,
) -> np.ndarray:
    """
    devuelve imagen sr como array numpy (h, w, c) uint8.
    """
    import torch
    from PIL import Image
    from torchvision import transforms

    dtype = torch.float16 if mixed_precision == "fp16" else torch.float32

    lr_img = Image.open(lr_path).convert("RGB")
    # upscale bicúbico como condición de entrada al unet
    hr_size = (lr_img.width * scale, lr_img.height * scale)
    lr_upscaled = lr_img.resize(hr_size, Image.BICUBIC)

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    lr_tensor = to_tensor(lr_upscaled).unsqueeze(0).to(device, dtype=dtype)

    pipe.scheduler.set_timesteps(sampler_steps)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(mixed_precision == "fp16")):
        latents = pipe.vae.encode(lr_tensor).latent_dist.sample() * 0.18215
        for t in pipe.scheduler.timesteps:
            noise_pred = pipe.unet(latents, t, encoder_hidden_states=None).sample
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        decoded = pipe.vae.decode(latents / 0.18215).sample

    # de [-1,1] a [0,255]
    decoded = (decoded.float().clamp(-1, 1) + 1) / 2
    decoded = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
    decoded = (decoded * 255).astype(np.uint8)
    return decoded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="inferencia sr con stablesr.")
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


def main() -> None:
    args = parse_args()
    log_file = Path("results/logs") / f"{args.exp_name}.log"
    setup_logging(log_file, args.log_level)
    fix_seed(args.seed)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("INFERENCIA SR")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    import torch
    device = check_cuda() if args.device == "cuda" else torch.device("cpu")

    out_dir = args.output_dir / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        logger.error(f"checkpoint no encontrado: {args.checkpoint}")
        sys.exit(1)

    pipe = load_model(args.checkpoint, args.mixed_precision, device)

    lr_paths = sorted(p for p in args.lr_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS)
    if not lr_paths:
        logger.error(f"no hay imágenes lr en {args.lr_dir}")
        sys.exit(1)

    logger.info(f"imágenes lr: {len(lr_paths)}")
    processed, skipped = 0, 0

    from PIL import Image
    import cv2

    for lr_path in tqdm(lr_paths, desc="inferencia sr", unit="img"):
        try:
            sr_array = run_inference_single(
                pipe, lr_path, args.scale, args.sampler_steps, device, args.mixed_precision
            )
            # quita el sufijo _xN del nombre para que coincida con el hr
            stem = lr_path.stem
            if stem.endswith(f"_x{args.scale}"):
                stem = stem[: -len(f"_x{args.scale}")]
            out_path = out_dir / f"{stem}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(sr_array, cv2.COLOR_RGB2BGR))
            processed += 1
            logger.debug(f"  guardado: {out_path}")
        except Exception as e:
            logger.warning(f"error en {lr_path.name}: {e}")
            skipped += 1

    logger.info(f"listo: {processed} imágenes sr, {skipped} omitidas")
    logger.info(f"resultados en: {out_dir}")


if __name__ == "__main__":
    main()
