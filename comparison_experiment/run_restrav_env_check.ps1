$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $PSScriptRoot ".venv_restrav\Scripts\python.exe"

if (-not (Test-Path $Py)) {
    throw "ReStraV venv not found: $Py"
}

$Code = @'
import sys
import torch
import torchvision
import torchcodec
from torchcodec.decoders import VideoDecoder
import numpy, pandas, sklearn, scipy, h5py

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("torchcodec", getattr(torchcodec, "__version__", "unknown"))
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("sklearn", sklearn.__version__)
print("scipy", scipy.__version__)
print("h5py", h5py.__version__)
print("ReStraV env check: OK")
'@

$Code | & $Py -
