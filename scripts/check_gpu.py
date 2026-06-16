"""Check PyTorch CUDA availability for the final project environment."""
import torch

print(f"torch: {torch.__version__}")
print(f"torch cuda runtime: {torch.version.cuda}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"test tensor: {tuple(y.shape)} on {y.device}")
    print(f"max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB")
