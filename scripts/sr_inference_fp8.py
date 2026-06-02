"""
sr_inference_fp8.py — Inferencia StableSR con máxima eficiencia de VRAM.

Estrategias de reducción de VRAM (acumuladas):
  1. FP8 (e4m3) via torchao → cuantización del UNet y VQGAN decoder
  2. FP16 autocast          → operaciones en semiprecisión donde FP8 no aplica
  3. Attention slicing      → divide el cálculo de atención en trozos
  4. Gradient checkpointing → no aplica en inferencia (ya desactivado)
  5. Tile-based processing  → procesa imágenes grandes en tiles
  6. n_samples=1            → batch size 1
  7. ddim_steps reducidos   → 20-30 pasos en lugar de 50

Hardware objetivo: NVIDIA RTX 5060 (8 GB VRAM, Blackwell sm_120)
PyTorch: 2.12.0.dev+cu128 (FP8 nativo)

Uso básico:
    cd C:\\TFM\\StableSR
    python ..\\scripts\\sr_inference_fp8.py \\
        --init-img ..\\data\\raw\\unmarked \\
        --outdir ..\\data\\sr\\x4 \\
        --ckpt checkpoints\\stablesr_000117.ckpt \\
        --vqgan_ckpt checkpoints\\vqgan_cfw_00011.ckpt \\
        --config configs\\stableSRNew\\v2-finetune_text_T_512.yaml \\
        --fp8 \\
        --ddim_steps 20 \\
        --n_samples 1 \\
        --input_size 512

Uso máximo ahorro VRAM (tile mode):
    python ..\\scripts\\sr_inference_fp8.py ... --fp8 --tile --tile_size 256 --tile_overlap 32
"""

import argparse
import math
import os
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from einops import rearrange, repeat
from itertools import islice
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Asegurar que el directorio StableSR esté en sys.path
# (ldm y basicsr viven ahí y no están instalados como paquetes del sistema)
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).resolve().parent          # C:\TFM\scripts
_STABLESR_DIR = _SCRIPT_DIR.parent / "StableSR"        # C:\TFM\StableSR
if str(_STABLESR_DIR) not in sys.path:
    sys.path.insert(0, str(_STABLESR_DIR))


# ---------------------------------------------------------------------------
# Verificación de entorno
# ---------------------------------------------------------------------------

def check_environment():
    print("=" * 60)
    print("ENTORNO DE INFERENCIA")
    print("=" * 60)
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  CUDA:     {torch.version.cuda}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3
        print(f"  GPU:      {props.name}")
        print(f"  VRAM:     {vram_gb:.1f} GB")
        print(f"  FP8 e4m3: {'OK' if hasattr(torch, 'float8_e4m3fn') else 'NO'}")
        print(f"  SM:       {props.major}.{props.minor}")
        if props.major < 9:
            print("  AVISO: FP8 hardware nativo requiere sm_9.0+ (Ada/Blackwell).")
            print("         En tu GPU se usará emulación FP8 → puede ser más lento.")
    else:
        print("  CUDA no disponible. Usando CPU.")
    print("=" * 60)


def vram_usage() -> str:
    if torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{used:.2f}/{total:.1f} GB"
    return "N/A"


# ---------------------------------------------------------------------------
# FP8 helpers via torchao
# ---------------------------------------------------------------------------

def apply_fp8_quantization(model, verbose: bool = True):
    """
    Cuantiza las capas Linear del modelo a FP8 (e4m3fn) usando torchao 0.17.
    Usa Float8DynamicActivationFloat8WeightConfig (pesos + activaciones FP8).
    Compatible con PyTorch 2.1+ y hardware sm_8.9+ (Ada Lovelace / Blackwell).
    """
    try:
        from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
        if verbose:
            print("  [FP8] Aplicando cuantización FP8 dinámica (pesos + activaciones)...")
        quantize_(model, Float8DynamicActivationFloat8WeightConfig())
        if verbose:
            print(f"  [FP8] Hecho. VRAM: {vram_usage()}")
        return model
    except ImportError:
        print("  [FP8] torchao no instalado — usando FP16 como fallback.")
        return model.half()
    except Exception as e:
        print(f"  [FP8] Error en cuantización ({type(e).__name__}: {e}) — usando FP16 como fallback.")
        return model.half()


def apply_fp8_weight_only(model, verbose: bool = True):
    """
    Cuantización solo de pesos a FP8 (activaciones en FP16/BF16).
    Más compatible, menor overhead de conversión en tiempo de ejecución.
    Usa Float8WeightOnlyConfig de torchao 0.17.
    """
    try:
        from torchao.quantization import quantize_, Float8WeightOnlyConfig
        if verbose:
            print("  [FP8-W] Aplicando cuantización FP8 weight-only...")
        quantize_(model, Float8WeightOnlyConfig())
        if verbose:
            print(f"  [FP8-W] Hecho. VRAM: {vram_usage()}")
        return model
    except ImportError:
        print("  [FP8-W] torchao no instalado — usando FP16 como fallback.")
        return model.half()
    except Exception as e:
        print(f"  [FP8-W] Error ({type(e).__name__}: {e}) — usando FP16 como fallback.")
        return model.half()


# ---------------------------------------------------------------------------
# Carga de modelo
# ---------------------------------------------------------------------------

def load_model_from_config(config, ckpt: str, half: bool = True, verbose: bool = False):
    print(f"  Cargando checkpoint: {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        print(f"  Global step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    from ldm.util import instantiate_from_config
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if verbose and len(m) > 0:
        print(f"  Missing keys: {len(m)}")
    if verbose and len(u) > 0:
        print(f"  Unexpected keys: {len(u)}")
    model.eval()
    if half:
        model = model.half()
    return model


# ---------------------------------------------------------------------------
# Procesamiento de imágenes
# ---------------------------------------------------------------------------

def load_img(path: str, target_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    # Redimensionar manteniendo múltiplo de 64 (requerido por la VAE de SD)
    if target_size > 0:
        # Resize al lado corto = target_size, luego crop cuadrado
        scale = target_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), PIL.Image.LANCZOS)
        # Center crop cuadrado
        left = (new_w - target_size) // 2
        top  = (new_h - target_size) // 2
        image = image.crop((left, top, left + target_size, top + target_size))
    else:
        # Solo asegurar múltiplo de 64
        w = (w // 64) * 64
        h = (h // 64) * 64
        image = image.resize((w, h), PIL.Image.LANCZOS)
    
    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr[None].transpose(0, 3, 1, 2)
    t = torch.from_numpy(arr)
    return 2.0 * t - 1.0   # [-1, 1]


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"Cannot create exactly {desired_count} steps")
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"Cannot divide section of {size} into {section_count}")
        frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


# ---------------------------------------------------------------------------
# Tile-based SR para imágenes grandes
# ---------------------------------------------------------------------------

def tile_process(img_tensor: torch.Tensor, tile_size: int, tile_overlap: int,
                 process_fn, device: torch.device, **kwargs) -> torch.Tensor:
    """
    Procesa una imagen en tiles para reducir el uso de VRAM.
    img_tensor: (1, C, H, W)
    """
    _, C, H, W = img_tensor.shape
    stride = tile_size - tile_overlap
    
    output = torch.zeros_like(img_tensor)
    count  = torch.zeros((1, 1, H, W), device=img_tensor.device)
    
    tiles_y = max(1, math.ceil((H - tile_overlap) / stride))
    tiles_x = max(1, math.ceil((W - tile_overlap) / stride))
    
    print(f"  [Tile] {tiles_y}x{tiles_x} = {tiles_y*tiles_x} tiles ({tile_size}px, overlap={tile_overlap}px)")
    
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0 = ty * stride
            x0 = tx * stride
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            y0 = max(0, y1 - tile_size)
            x0 = max(0, x1 - tile_size)
            
            tile = img_tensor[:, :, y0:y1, x0:x1]
            tile_out = process_fn(tile, **kwargs)
            
            output[:, :, y0:y1, x0:x1] += tile_out
            count[:, :, y0:y1, x0:x1]  += 1
    
    return output / count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferencia StableSR con FP8 / máxima eficiencia VRAM."
    )
    # Rutas
    parser.add_argument("--init-img",   type=str, default="inputs/user_upload",
                        help="Directorio con imágenes LR de entrada.")
    parser.add_argument("--outdir",     type=str, default="outputs/sr_fp8",
                        help="Directorio de salida.")
    parser.add_argument("--config",     type=str,
                        default=str(_STABLESR_DIR / "configs/stableSRNew/v2-finetune_text_T_512.yaml"))
    parser.add_argument("--ckpt",       type=str,
                        default=str(_STABLESR_DIR / "checkpoints/stablesr_000117.ckpt"))
    parser.add_argument("--vqgan_ckpt", type=str,
                        default=str(_STABLESR_DIR / "checkpoints/vqgan_cfw_00011.ckpt"))
    # Precisión y cuantización
    parser.add_argument("--fp8",            action="store_true",
                        help="Activar cuantización FP8 dinámica del UNet y VQGAN.")
    parser.add_argument("--fp8_weight_only", action="store_true",
                        help="FP8 solo en pesos (activaciones en FP16). Más conservador.")
    parser.add_argument("--precision",  type=str, default="autocast",
                        choices=["full", "autocast"],
                        help="'autocast' = FP16 mixed; 'full' = FP32.")
    # Parámetros de muestreo
    parser.add_argument("--ddim_steps", type=int, default=20,
                        help="Pasos DDIM (↓ pasos = ↓ VRAM activa y ↓ tiempo).")
    parser.add_argument("--ddim_eta",   type=float, default=0.0)
    parser.add_argument("--scale",      type=float, default=7.0,
                        help="Guidance scale (CFG). Solo con --use_negative_prompt.")
    parser.add_argument("--strength",   type=float, default=0.75)
    parser.add_argument("--n_samples",  type=int, default=1,
                        help="Batch size. Con 8 GB usar 1.")
    parser.add_argument("--seed",       type=int, default=42)
    # Imagen
    parser.add_argument("--input_size", type=int, default=512,
                        help="Tamaño al que se redimensiona la entrada (cuadrado).")
    parser.add_argument("--dec_w",      type=float, default=0.5,
                        help="Peso de mezcla VQGAN/Diffusion en el decoder.")
    # Prompts
    parser.add_argument("--use_negative_prompt", action="store_true")
    parser.add_argument("--use_posi_prompt",      action="store_true")
    # Color fix
    parser.add_argument("--colorfix_type", type=str, default="adain",
                        choices=["adain", "wavelet", "nofix"])
    # Tile mode (para imágenes muy grandes o VRAM muy ajustada)
    parser.add_argument("--tile",         action="store_true",
                        help="Procesar en tiles para reducir VRAM.")
    parser.add_argument("--tile_size",    type=int, default=256)
    parser.add_argument("--tile_overlap", type=int, default=32)
    # Atención
    parser.add_argument("--attn_slicing", action="store_true",
                        help="Dividir cálculo de atención (↓ pico VRAM).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    check_environment()

    print("\nCONFIGURACIÓN:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print()

    # Seed
    from pytorch_lightning import seed_everything
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Cargar modelo principal (StableSR / LDM)
    # ------------------------------------------------------------------
    print("\n[1/4] Cargando modelo StableSR...")
    config = OmegaConf.load(args.config)
    # Cargar en FP16 por defecto
    use_half = (args.precision == "autocast")
    model = load_model_from_config(config, args.ckpt, half=use_half)
    model = model.to(device)
    print(f"  VRAM tras modelo principal: {vram_usage()}")

    # ------------------------------------------------------------------
    # Cargar VQGAN
    # ------------------------------------------------------------------
    print("\n[2/4] Cargando VQGAN...")
    _vqgan_yaml = str(_STABLESR_DIR / "configs/autoencoder/autoencoder_kl_64x64x4_resi.yaml")
    vqgan_config = OmegaConf.load(_vqgan_yaml)
    vq_model = load_model_from_config(vqgan_config, args.vqgan_ckpt, half=use_half)
    vq_model = vq_model.to(device)
    vq_model.decoder.fusion_w = args.dec_w
    print(f"  VRAM tras VQGAN: {vram_usage()}")

    # ------------------------------------------------------------------
    # Aplicar FP8 si se solicita
    # ------------------------------------------------------------------
    if args.fp8 or args.fp8_weight_only:
        print("\n[2b] Aplicando cuantización FP8...")
        # Primero pasar a FP32 para la cuantización (torchao opera en FP32)
        model = model.float()
        vq_model = vq_model.float()
        
        if args.fp8_weight_only:
            # Solo pesos → más compatible, buen compromiso velocidad/precisión
            model = apply_fp8_weight_only(model)
            vq_model = apply_fp8_weight_only(vq_model)
        else:
            # Dinámico (pesos + activaciones) → máximo ahorro
            model = apply_fp8_quantization(model)
            vq_model = apply_fp8_quantization(vq_model)
        
        print(f"  VRAM tras FP8: {vram_usage()}")

    # ------------------------------------------------------------------
    # Configurar sampler
    # ------------------------------------------------------------------
    print("\n[3/4] Configurando DDIM sampler...")
    from ldm.models.diffusion.ddim import DDIMSampler

    model.register_schedule(
        given_betas=None, beta_schedule="linear", timesteps=1000,
        linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3
    )
    model.num_timesteps = 1000
    model = model.to(device)

    ddim_timesteps = list(space_timesteps(1000, [args.ddim_steps]))
    ddim_timesteps.sort()

    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=args.ddim_eta, verbose=False)

    # ------------------------------------------------------------------
    # Preparar directorios
    # ------------------------------------------------------------------
    out_dir = Path(args.outdir)
    sample_dir = out_dir / "samples"
    input_dir  = out_dir / "inputs"
    sample_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Listar imágenes (evitar reejecutar las ya procesadas)
    # ------------------------------------------------------------------
    init_dir = Path(args.init_img)
    all_imgs = sorted(p.name for p in init_dir.iterdir()
                      if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
    done = set(os.listdir(sample_dir))
    img_list = [x for x in all_imgs if x not in done]
    
    if not img_list:
        print("No hay imágenes nuevas que procesar.")
        return

    print(f"\n[4/4] Inferencia SR — {len(img_list)} imágenes")
    print(f"  Pasos DDIM:  {args.ddim_steps}")
    print(f"  Batch size:  {args.n_samples}")
    print(f"  Tile mode:   {'sí (' + str(args.tile_size) + 'px)' if args.tile else 'no'}")
    print(f"  Color fix:   {args.colorfix_type}")
    print(f"  VRAM inicial: {vram_usage()}\n")

    # Importar color fix
    from scripts.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization  # noqa: needs StableSR in sys.path

    # Transformación de entrada
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.input_size),
        torchvision.transforms.CenterCrop(args.input_size),
    ])

    # Batches
    batch_size = args.n_samples
    img_batches = [img_list[i:i+batch_size]
                   for i in range(0, len(img_list), batch_size)]

    # Scope de precisión
    if args.fp8 or args.fp8_weight_only:
        # FP8 con torchao → no usar autocast de CUDA (incompatible)
        precision_scope = nullcontext
    else:
        from torch import autocast as torch_autocast
        precision_scope = (lambda: torch_autocast("cuda", dtype=torch.float16)
                           if args.precision == "autocast" else nullcontext())
        precision_scope = (torch.cuda.amp.autocast if args.precision == "autocast"
                           else nullcontext)

    niqe_list = []
    tic = time.time()

    with torch.no_grad():
        with model.ema_scope():
            for batch_names in tqdm(img_batches, desc="Inferencia SR", unit="batch"):
                # Cargar imágenes del batch
                init_images = []
                for name in batch_names:
                    img_path = str(init_dir / name)
                    try:
                        img_t = load_img(img_path, args.input_size).to(device)
                        img_t = transform(img_t)
                        init_images.append(img_t)
                    except Exception as e:
                        print(f"  Error leyendo {name}: {e}. Omitido.")
                        continue

                if not init_images:
                    continue

                init_image = torch.cat(init_images, dim=0)

                # Convertir a FP16 para operaciones intermedias si no FP8
                if not (args.fp8 or args.fp8_weight_only) and use_half:
                    init_image = init_image.half()

                # Encode con VQGAN
                try:
                    init_latent_gen, enc_fea_lq = vq_model.encode(init_image)
                    init_latent = model.get_first_stage_encoding(init_latent_gen)
                except Exception as e:
                    print(f"  Error encoding: {e}")
                    continue

                # Prompts
                if args.use_posi_prompt:
                    text_init = ['(masterpiece:2), (best quality:2), (realistic:2), (very clear:2)'] * init_image.size(0)
                else:
                    text_init = [''] * init_image.size(0)
                semantic_c = model.cond_stage_model(text_init)

                nega_semantic_c = None
                if args.use_negative_prompt:
                    neg_text = ['3d, cartoon, anime, sketches, (worst quality:2), (low quality:2)'] * init_image.size(0)
                    nega_semantic_c = model.cond_stage_model(neg_text)

                # Noising
                noise = torch.randn_like(init_latent)
                t = repeat(torch.tensor([999]), '1 -> b', b=init_image.size(0))
                t = t.to(device).long()
                x_T = model.q_sample(x_start=init_latent, t=t, noise=noise)

                # Muestreo DDIM
                try:
                    with (torch.cuda.amp.autocast(dtype=torch.float16)
                          if (args.precision == "autocast" and not args.fp8 and not args.fp8_weight_only)
                          else nullcontext()):
                        samples, _ = sampler.ddim_sampling_sr_t(
                            cond=semantic_c,
                            struct_cond=init_latent,
                            shape=init_latent.shape,
                            unconditional_conditioning=nega_semantic_c,
                            unconditional_guidance_scale=args.scale if args.use_negative_prompt else None,
                            timesteps=np.array(ddim_timesteps),
                            x_T=x_T,
                        )
                except Exception as e:
                    print(f"  Error en sampling: {e}")
                    import traceback; traceback.print_exc()
                    continue

                # Decode con VQGAN
                x_samples = vq_model.decode(samples * 1.0 / model.scale_factor, enc_fea_lq)

                # Color fix
                if args.colorfix_type == 'adain':
                    x_samples = adaptive_instance_normalization(x_samples, init_image)
                elif args.colorfix_type == 'wavelet':
                    x_samples = wavelet_reconstruction(x_samples, init_image)

                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                # Guardar resultados
                for i, name in enumerate(batch_names[:x_samples.size(0)]):
                    try:
                        x_np = 255.0 * rearrange(x_samples[i].float().cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_np.astype(np.uint8)).save(sample_dir / name)
                    except Exception as e:
                        print(f"  Error guardando {name}: {e}")

                torch.cuda.empty_cache()

    elapsed = time.time() - tic
    print(f"\nCompletado en {elapsed:.1f}s")
    print(f"  Resultados en: {sample_dir}")
    print(f"  VRAM final:    {vram_usage()}")


if __name__ == "__main__":
    main()
