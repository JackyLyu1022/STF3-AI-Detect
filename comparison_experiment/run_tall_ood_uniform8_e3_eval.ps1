$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
$Py = Join-Path $ScriptDir ".venv_restrav\Scripts\python.exe"
$Checkpoint = Join-Path $ProjectRoot "comparison_experiment\results\TALL\ood_tall_uniform8_e5\best.pt"

if (-not (Test-Path $Py)) {
    throw "Shared comparison venv not found: $Py"
}

if (-not (Test-Path $Checkpoint)) {
    throw "TALL checkpoint not found: $Checkpoint"
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
  --out-dir "comparison_experiment/results/TALL/ood_tall_uniform8_e3_eval" `
  --checkpoint "comparison_experiment/results/TALL/ood_tall_uniform8_e5/best.pt" `
  --eval-only `
  --epochs 3 `
  --batch-size 2 `
  --lr 1.5e-5 `
  --weight-decay 1e-5 `
  --num-frames 8 `
  --frame-size 112 `
  --thumbnail-rows 2 `
  --device "cuda:0" `
  --workers 0 `
  --amp
