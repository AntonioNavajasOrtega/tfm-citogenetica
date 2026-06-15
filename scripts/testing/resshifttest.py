# test_resshift.py
import torch
from omegaconf import OmegaConf
import xformers

print(f"Torch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability()}")  # debe ser (12, 0)
print(f"xformers: {xformers.__version__}")

# Simula lo que hace run_resshift.py y train_resshift.py
import sys
sys.path.insert(0, r"c:\TFM\ResShift")
from sampler import ResShiftSampler
from utils.util_common import get_obj_from_str
print("ResShift imports OK")