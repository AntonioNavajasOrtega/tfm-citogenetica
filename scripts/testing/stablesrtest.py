# test_stablesr.py
import torch
from omegaconf import OmegaConf
import xformers
import clip

print(f"Torch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability()}")  # debe ser (12, 0)
print(f"xformers: {xformers.__version__}")

# Simula lo que hace run_stablesr.py (subprocess, pero verifica que los scripts existen)
from pathlib import Path
stablesr_dir = Path(r"c:\TFM\StableSR")
for script in ["sr_val_ddim_text_T_negativeprompt.py", "sr_val_ddpm_text_T_vqganfin_old.py"]:
    p = stablesr_dir / "scripts" / script
    print(f"{'OK' if p.exists() else 'FALTA'}: {script}")

print("StableSR imports OK")