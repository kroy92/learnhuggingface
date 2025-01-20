import torch

if torch.xpu.is_available():
    print(f"Number of XPUs available: {torch.xpu.device_count()}")
else:
    print("No XPU found. Using CPU.")
    device = torch.device("cpu")