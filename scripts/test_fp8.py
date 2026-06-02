"""Test FP8 en RTX 5060 con torchao 0.17"""
import torch
import torch.nn as nn
import torchao

print(f"torchao: {torchao.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"GPU:     {torch.cuda.get_device_name(0)}")
vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"VRAM:    {vram_total:.1f} GB")
print()

from torchao.quantization import quantize_, Float8WeightOnlyConfig, Float8DynamicActivationFloat8WeightConfig

x = torch.randn(2, 256, device="cuda")

# Test 1: FP8 weight-only (recomendado para StableSR)
model_wo = nn.Sequential(
    nn.Linear(256, 256), nn.SiLU(),
    nn.Linear(256, 256), nn.SiLU(),
    nn.Linear(256, 128)
).cuda()
quantize_(model_wo, Float8WeightOnlyConfig())
with torch.no_grad():
    y = model_wo(x)
vram_used = torch.cuda.memory_allocated() / 1024**3
print(f"[OK] FP8 weight-only     — output: {y.shape} | VRAM: {vram_used:.3f} GB")

torch.cuda.empty_cache()

# Test 2: FP8 dinámico (pesos + activaciones)
model_dyn = nn.Sequential(
    nn.Linear(256, 256), nn.SiLU(),
    nn.Linear(256, 256), nn.SiLU(),
    nn.Linear(256, 128)
).cuda()
quantize_(model_dyn, Float8DynamicActivationFloat8WeightConfig())
with torch.no_grad():
    y2 = model_dyn(x)
vram_used2 = torch.cuda.memory_allocated() / 1024**3
print(f"[OK] FP8 dinámico        — output: {y2.shape} | VRAM: {vram_used2:.3f} GB")

torch.cuda.empty_cache()

# Comparación: FP16 baseline
model_fp16 = nn.Sequential(
    nn.Linear(256, 256), nn.SiLU(),
    nn.Linear(256, 256), nn.SiLU(),
    nn.Linear(256, 128)
).cuda().half()
with torch.no_grad():
    y3 = model_fp16(x.half())
vram_fp16 = torch.cuda.memory_allocated() / 1024**3
print(f"[OK] FP16 baseline       — output: {y3.shape} | VRAM: {vram_fp16:.3f} GB")

print()
print("=" * 50)
print("FP8 LISTO para inferencia StableSR en RTX 5060")
print("=" * 50)
