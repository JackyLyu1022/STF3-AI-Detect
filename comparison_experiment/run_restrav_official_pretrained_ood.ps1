$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
$Py = Join-Path $ScriptDir ".venv_restrav\Scripts\python.exe"
$PretrainedDir = Join-Path $ScriptDir "ReStraV\pretrained"

if (-not (Test-Path $Py)) {
    throw "ReStraV venv not found: $Py"
}

if (-not (Test-Path (Join-Path $PretrainedDir "model.pt"))) {
    throw @"
Official ReStraV pretrained files are not found.

Please put these files into:
  $PretrainedDir

Required:
  model.pt
  mean.npy
  std.npy

Optional:
  best_tau.npy

Then rerun this script.
"@
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

& $Py (Join-Path $ScriptDir "run_restrav_ood_compare.py") `
  --project-root $ProjectRoot `
  --restrav-root "comparison_experiment/ReStraV" `
  --data-root "data/GenVideo-Val" `
  --val-csv "data/GenVideo-Val/splits/ood_val.csv" `
  --test-csv "data/GenVideo-Val/splits/ood_test.csv" `
  --out-dir "comparison_experiment/results/ReStraV/ood_official_pretrained_dinov2_t24_w2" `
  --feature-cache-dir "comparison_experiment/results/ReStraV/ood_dinov2_t24_w2_e5/features" `
  --device "cuda:0" `
  --num-frames 24 `
  --window-sec 2.0 `
  --batch-size 32 `
  --eval-only `
  --pretrained-dir "comparison_experiment/ReStraV/pretrained" `
  --pretrained-positive-class "real"
