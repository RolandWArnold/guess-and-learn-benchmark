#!/usr/bin/env python

import platform
import psutil
import torch

try:
    import GPUtil

    gpu_available = True
except ImportError:
    gpu_available = False


def get_system_specs():
    """Gathers and prints key hardware and software specifications."""

    print("--- System Specifications ---")

    # OS Info
    print(f"Operating System: {platform.system()} {platform.release()}")

    # CPU Info
    print(f"CPU Model: {platform.processor()}")
    print(f"Physical Cores: {psutil.cpu_count(logical=False)}")
    print(f"Total Cores: {psutil.cpu_count(logical=True)}")

    # RAM Info
    ram_info = psutil.virtual_memory()
    print(f"Total RAM: {ram_info.total / (1024**3):.2f} GB")

    print("\n--- Software Specifications ---")
    print(f"Python Version: {platform.python_version()}")

    # PyTorch and CUDA Info
    if "torch" in globals():
        print(f"PyTorch Version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA Version (available to PyTorch): {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        else:
            print("CUDA: Not available to PyTorch")

    # GPU Info
    if gpu_available and torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        if gpus:
            print("\n--- GPU Specifications ---")
            for i, gpu in enumerate(gpus):
                print(f"GPU {i}:")
                print(f"  Model: {gpu.name}")
                print(f"  VRAM: {gpu.memoryTotal} MB")
                print(f"  Driver Version: {gpu.driver}")
        else:
            print("\n--- GPU Specifications ---")
            print("GPUtil found no GPUs.")
    else:
        print("\n--- GPU Specifications ---")
        print("GPU information not available (GPUtil not installed or CUDA not available).")


if __name__ == "__main__":
    get_system_specs()
