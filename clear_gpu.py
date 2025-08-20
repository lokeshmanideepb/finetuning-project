import torch
torch.cuda.empty_cache()
if torch.cuda.is_available():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("ERROR: CUDA is not available to PyTorch.")