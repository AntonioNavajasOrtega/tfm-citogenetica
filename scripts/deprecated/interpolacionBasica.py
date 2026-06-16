"""
Escala imágenes de data/test/images a x2 y x4 usando interpolación bicúbica.
Salida: data/test/images_x2/ y data/test/images_x4/
"""

import os
from pathlib import Path
from PIL import Image

INPUT_DIR  = Path("data/test/images")
OUTPUT_X2  = Path("data/test/images_x2_interpolacion")
OUTPUT_X4  = Path("data/test/images_x4_interpolacion")
EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def upscale(img: Image.Image, scale: int) -> Image.Image:
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.BICUBIC)

def main():
    OUTPUT_X2.mkdir(parents=True, exist_ok=True)
    OUTPUT_X4.mkdir(parents=True, exist_ok=True)

    images = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in EXTENSIONS]
    if not images:
        print(f"No se encontraron imágenes en {INPUT_DIR}")
        return

    for path in sorted(images):
        img = Image.open(path)
        print(f"[{img.size}] {path.name}", end=" → ")

        upscale(img, 2).save(OUTPUT_X2 / path.name)
        upscale(img, 4).save(OUTPUT_X4 / path.name)

        print(f"x2 {tuple(s*2 for s in img.size)}, x4 {tuple(s*4 for s in img.size)}")

    print(f"\nListo. {len(images)} imágenes procesadas.")
    print(f"  x2 → {OUTPUT_X2}")
    print(f"  x4 → {OUTPUT_X4}")

if __name__ == "__main__":
    main()