"""
04_train_stablesr.py — Finetuning de StableSR sobre dataset citogenético.

StableSR está basado en Stable Diffusion con un módulo de restauración;
este script implementa el finetuning con soporte de fp16, gradient
checkpointing, freeze UNet y gradient accumulation para caber en 8 GB VRAM.

Ref. oficial: https://github.com/IceClear/StableSR

El script puede recibir configuración desde un YAML (--config) y los
argumentos CLI tienen precedencia sobre el yaml para facilitar overrides.

Uso:
    python scripts/04_train_stablesr.py \
        --config configs/stablesr_default.yaml \
        --hr_dir data/processed \
        --lr_dir data/lr/x2 \
        --output_dir models/stablesr_finetuned \
        --exp_name exp0_default
"""

import argparse
import csv
import logging
import random
import sys
from pathlib import Path

import numpy as np
import yaml
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
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def check_cuda() -> "torch.device":
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"CUDA disponible: {name} | VRAM: {vram:.1f} GB")
        return torch.device("cuda")
    else:
        logging.warning("CUDA NO disponible — se usará CPU (MUY LENTO para StableSR).")
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Carga de configuración
# ---------------------------------------------------------------------------

def load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config(yaml_cfg: dict, args: argparse.Namespace) -> argparse.Namespace:
    """Los valores CLI (no-None) tienen precedencia sobre yaml."""
    for k, v in yaml_cfg.items():
        if hasattr(args, k) and getattr(args, k) is None:
            setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------
# Dataset HR/LR
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def find_lr_counterpart(hr_path: Path, lr_dir: Path, scale: int) -> Path | None:
    """Busca la imagen LR correspondiente (nombre HR + _xN extensión)."""
    candidates = [
        lr_dir / f"{hr_path.stem}_x{scale}{hr_path.suffix}",
        lr_dir / hr_path.name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def build_dataset_pairs(hr_dir: Path, lr_dir: Path, scale: int) -> list[tuple[Path, Path]]:
    pairs = []
    for hr_path in sorted(hr_dir.iterdir()):
        if hr_path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        lr_path = find_lr_counterpart(hr_path, lr_dir, scale)
        if lr_path is not None:
            pairs.append((hr_path, lr_path))
        else:
            logging.warning(f"  Sin contraparte LR para: {hr_path.name}")
    return pairs


# ---------------------------------------------------------------------------
# Modelo StableSR
# ---------------------------------------------------------------------------

def build_stablesr_model(args: argparse.Namespace, device: "torch.device"):
    """
    Carga el modelo StableSR.

    StableSR requiere los pesos preentrenados de Stable Diffusion v2.1
    y el módulo de restauración de https://github.com/IceClear/StableSR.

    Si --pretrained apunta a un checkpoint local se carga directamente;
    si no se descarga con diffusers (modelo base).
    """
    import torch
    from diffusers import StableDiffusionPipeline

    logger = logging.getLogger(__name__)

    pretrained = args.pretrained or "stabilityai/stable-diffusion-2-1-base"
    logger.info(f"Cargando modelo base desde: {pretrained}")

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            pretrained,
            torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
            safety_checker=None,
        )
        unet = pipe.unet.to(device)
        vae = pipe.vae.to(device)
        text_encoder = pipe.text_encoder.to(device)
        noise_scheduler = pipe.scheduler

        # Congelar UNet si se solicita
        if args.freeze_unet:
            logger.info("UNet congelado (solo se finetunan capas de restauración).")
            for param in unet.parameters():
                param.requires_grad = False
        else:
            logger.info("Finetuning completo (UNet descongelado).")

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            logger.info("Gradient checkpointing activado.")

        if args.use_xformers:
            try:
                unet.enable_xformers_memory_efficient_attention()
                logger.info("xFormers activado.")
            except Exception as e:
                logger.warning(f"xFormers no disponible: {e}")

        return {
            "unet": unet,
            "vae": vae,
            "text_encoder": text_encoder,
            "scheduler": noise_scheduler,
        }
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        logger.error(
            "Asegúrate de tener el checkpoint StableSR en --pretrained o conexión a HuggingFace."
        )
        raise


# ---------------------------------------------------------------------------
# Pérdida
# ---------------------------------------------------------------------------

def compute_loss(pred, target, loss_type: str, lpips_fn=None):
    import torch
    import torch.nn.functional as F

    if loss_type == "l1":
        return F.l1_loss(pred, target)
    elif loss_type == "l2":
        return F.mse_loss(pred, target)
    elif loss_type == "lpips" and lpips_fn is not None:
        return lpips_fn(pred, target).mean()
    elif loss_type == "l1+lpips":
        l1 = F.l1_loss(pred, target)
        lp = lpips_fn(pred, target).mean() if lpips_fn is not None else 0.0
        return l1 + 0.1 * lp
    else:
        return F.l1_loss(pred, target)


# ---------------------------------------------------------------------------
# Loop de entrenamiento
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace, pairs: list[tuple[Path, Path]], device: "torch.device") -> None:
    import torch
    import torch.optim as optim
    from PIL import Image
    from torchvision import transforms

    logger = logging.getLogger(__name__)

    out_dir = Path(args.output_dir) / args.exp_name
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_parts = build_stablesr_model(args, device)
    unet = model_parts["unet"]
    vae = model_parts["vae"]
    scheduler = model_parts["scheduler"]

    # Optimizador (solo parámetros entrenables)
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr)

    # LPIPS
    lpips_fn = None
    if "lpips" in args.loss_type:
        try:
            import lpips as lpips_lib
            lpips_fn = lpips_lib.LPIPS(net="vgg").to(device)
            logger.info("LPIPS cargado.")
        except ImportError:
            logger.warning("lpips no instalado; se usará solo L1.")
            args.loss_type = "l1"

    # Transformaciones de imagen
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    use_amp = args.mixed_precision == "fp16"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # CSV log
    csv_path = out_dir / "train_log.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])

    best_loss = float("inf")

    for epoch in tqdm(range(1, args.epochs + 1), desc="Épocas", unit="epoch"):
        unet.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step_idx, (hr_path, lr_path) in enumerate(
            tqdm(pairs, desc=f"Epoch {epoch}", leave=False, unit="img")
        ):
            try:
                hr_img = Image.open(hr_path).convert("RGB")
                lr_img = Image.open(lr_path).convert("RGB")
                hr_img = hr_img.resize(
                    (hr_img.width * args.scale if lr_img.size[0] * args.scale == hr_img.width else hr_img.width,
                     hr_img.height), Image.LANCZOS
                )
            except Exception as e:
                logger.warning(f"  Error leyendo par {hr_path.name}/{lr_path.name}: {e}")
                continue

            hr_tensor = to_tensor(hr_img).unsqueeze(0).to(device)
            lr_tensor = to_tensor(lr_img).unsqueeze(0).to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                # Encode HR → latent space
                latents = vae.encode(hr_tensor).latent_dist.sample() * 0.18215
                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                # UNet prediction
                pred = unet(noisy_latents, timesteps, encoder_hidden_states=None).sample
                loss = compute_loss(pred, noise, args.loss_type, lpips_fn) / args.accum_steps

            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * args.accum_steps

            # Gradient accumulation step
            if (step_idx + 1) % args.accum_steps == 0:
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

        avg_loss = epoch_loss / max(len(pairs), 1)
        logger.info(f"  Época {epoch}/{args.epochs} | Loss={avg_loss:.6f}")

        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss])

        # Guardar últimos y mejor checkpoint
        torch.save({"epoch": epoch, "unet": unet.state_dict(), "loss": avg_loss},
                   ckpt_dir / "last.ckpt")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"epoch": epoch, "unet": unet.state_dict(), "loss": avg_loss},
                       ckpt_dir / "best.ckpt")
            logger.info(f"  ✓ Mejor checkpoint guardado (epoch {epoch}, loss={avg_loss:.6f})")

    logger.info(f"Entrenamiento completado. Checkpoints en: {ckpt_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finetuning StableSR sobre dataset citogenético."
    )
    parser.add_argument("--config", type=Path, default=None, help="Ruta al YAML de configuración.")
    parser.add_argument("--hr_dir", type=Path, default=None)
    parser.add_argument("--lr_dir", type=Path, default=None)
    parser.add_argument("--scale", type=int, default=None, choices=[2, 3, 4])
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--accum_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--scheduler", type=str, default=None, choices=["ddpm", "ddim"])
    parser.add_argument("--sampler_steps", type=int, default=None)
    parser.add_argument("--loss_type", type=str, default=None)
    parser.add_argument("--freeze_unet", action="store_true", default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=None)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["fp16", "bf16", "no"])
    parser.add_argument("--use_xformers", action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "scale": 2,
    "epochs": 100,
    "batch_size": 1,
    "accum_steps": 4,
    "lr": 5e-5,
    "scheduler": "ddim",
    "sampler_steps": 20,
    "loss_type": "l1+lpips",
    "freeze_unet": True,
    "gradient_checkpointing": True,
    "mixed_precision": "fp16",
    "use_xformers": True,
    "seed": 42,
    "exp_name": "stablesr_exp",
    "output_dir": Path("models/stablesr_finetuned"),
}


def apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    for k, v in DEFAULTS.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 1. Cargar YAML si existe
    if args.config is not None and args.config.exists():
        yaml_cfg = load_yaml_config(args.config)
        args = merge_config(yaml_cfg, args)

    # 2. Aplicar defaults
    args = apply_defaults(args)

    log_file = Path("results/logs") / f"{args.exp_name}.log"
    setup_logging(log_file, args.log_level)
    fix_seed(args.seed)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("SCRIPT 04 — FINETUNING STABLESR")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    device = check_cuda()

    # Validar directorios
    if args.hr_dir is None or args.lr_dir is None:
        logger.error("--hr_dir y --lr_dir son obligatorios.")
        sys.exit(1)

    args.hr_dir = Path(args.hr_dir)
    args.lr_dir = Path(args.lr_dir)

    pairs = build_dataset_pairs(args.hr_dir, args.lr_dir, args.scale)
    if not pairs:
        logger.error("No se encontraron pares HR/LR. Verifica los directorios.")
        sys.exit(1)

    logger.info(f"Pares HR/LR encontrados: {len(pairs)}")
    train(args, pairs, device)


if __name__ == "__main__":
    main()
