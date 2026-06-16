$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
$Py = Join-Path $ScriptDir ".venv_restrav\Scripts\python.exe"

if (-not (Test-Path $Py)) {
    throw "ReStraV venv not found: $Py"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

& $Py (Join-Path $ScriptDir "run_restrav_ood_compare.py") `
  --project-root $ProjectRoot `
  --restrav-root "comparison_experiment/ReStraV" `
  --data-root "data/GenVideo-Val" `
  --train-csv "data/GenVideo-Val/splits/ood_train.csv" `
  --val-csv "data/GenVideo-Val/splits/ood_val.csv" `
  --test-csv "data/GenVideo-Val/splits/ood_test.csv" `
  --out-dir "comparison_experiment/results/ReStraV/ood_dinov2_t24_w2_e5" `
  --device "cuda:0" `
  --num-frames 24 `
  --window-sec 2.0 `
  --epochs 5 `
  --batch-size 32 `
  --lr 0.001 `
  --pos-weight-mode "auto"
