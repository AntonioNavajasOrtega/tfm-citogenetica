"""
download_sr_model.py — descarga y cachea el modelo stable-diffusion-x4-upscaler.

ejecutar una vez antes de run_sr.py para tener el modelo listo en local.
la descarga pesa ~5GB y va al caché de huggingface.

uso:
    python scripts/download_sr_model.py
"""

import sys

def main() -> None:
    try:
        import torch
        from diffusers import StableDiffusionUpscalePipeline
    except ImportError as e:
        print(f"falta dependencia: {e}")
        print("instala con: pip install diffusers transformers accelerate")
        sys.exit(1)

    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    print(f"descargando {model_id} ...")
    print("tamaño aprox: 5 GB — puede tardar varios minutos según conexión\n")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    print("\n✓ descarga completada")
    print("el modelo queda cacheado en el directorio de huggingface")
    print("ya puedes ejecutar: python scripts/run_sr.py")

if __name__ == "__main__":
    main()
