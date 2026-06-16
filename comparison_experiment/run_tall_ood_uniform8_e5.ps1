$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
$Py = Join-Path $ScriptDir ".venv_restrav\Scripts\python.exe"

if (-not (Test-Path $Py)) {
    throw "Shared comparison venv not found: $Py"
}

if (-not (Test-Path (Join-Path $ScriptDir "TALL"))) {
    throw "TALL repo not found under comparison_experiment\TALL"
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

& $Py (Join-Path $ScriptDir "run_tall_ood_compare.py") `
  --project-root $ProjectRoot `
  --tall-root "comparison_experiment/TALL" `
  --data-root "data/GenVideo-Val" `
  --train-csv "data/GenVideo-Val/splits/ood_train.csv" `
  --val-csv "data/GenVideo-Val/splits/ood_val.csv" `
  --test-csv "data/GenVideo-Val/splits/ood_test.csv" `
  --out-dir "comparison_experiment/results/TALL/ood_tall_uniform8_e5" `
  --pretrained `
  --epochs 5 `
  --batch-size 2 `
  --lr 1.5e-5 `
  --weight-decay 1e-5 `
  --num-frames 8 `
  --frame-size 112 `
  --thumbnail-rows 2 `
  --device "cuda:0" `
  --workers 0 `
  --amp
