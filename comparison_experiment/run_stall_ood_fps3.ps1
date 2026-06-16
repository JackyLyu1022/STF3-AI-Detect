$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
$Py = Join-Path $ScriptDir ".venv_restrav\Scripts\python.exe"
$DinoWeights = Join-Path $ScriptDir "STALL\dinov3\weights\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

if (-not (Test-Path $Py)) {
    throw "Shared ReStraV/STALL venv not found: $Py"
}

if (-not (Test-Path $DinoWeights)) {
    throw @"
DINOv3 weights are not found.

Please download:
  dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

and put it here:
  $DinoWeights

Official download page:
  https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
"@
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

& $Py (Join-Path $ScriptDir "run_stall_ood_compare.py") `
  --project-root $ProjectRoot `
  --stall-root "comparison_experiment/STALL" `
  --data-root "data/GenVideo-Val" `
  --val-csv "data/GenVideo-Val/splits/ood_val.csv" `
  --test-csv "data/GenVideo-Val/splits/ood_test.csv" `
  --out-dir "comparison_experiment/results/STALL/ood_vatex_dinov3_fps3_d1" `
  --params "comparison_experiment/STALL/precomputed/stall_params_vatex_dino_v3.npz" `
  --dino-repo "comparison_experiment/STALL/dinov3" `
  --dino-weights "comparison_experiment/STALL/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth" `
  --sampling-mode "official_fps" `
  --device "cuda:0" `
  --duration 1 `
  --target-fps 3 `
  --workers 4 `
  --video-batch 4 `
  --frame-batch 8
