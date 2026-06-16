Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "D:\VsCode Program\Python\content_security\final_project"

$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
  $Python = "python"
}

$Modes = @(
  "spatial_dino",
  "frequency_wave",
  "temporal_d3_restrav",
  "spatial_frequency_new",
  "spatial_temporal_new",
  "stf3_new_concat",
  "stf3_new"
)

foreach ($Mode in $Modes) {
  $RunDir = "runs\ood_$Mode"
  $OutDir = "outputs\ood_$Mode"
  Write-Host "========== OOD ablation: $Mode ==========" -ForegroundColor Cyan

  & $Python -m src.train `
    --model $Mode `
    --train-csv data\GenVideo-Val\splits\ood_train.csv `
    --val-csv data\GenVideo-Val\splits\ood_val.csv `
    --epochs 10 `
    --batch-size 1 `
    --num-frames 8 `
    --image-size 112 `
    --foundation-backbone dinov2_vits14 `
    --wavelet-aug-prob 0.1 `
    --branch-dropout 0.1 `
    --aux-loss-weight 0.2 `
    --amp `
    --out-dir $RunDir

  & $Python -m src.evaluate `
    --checkpoint "$RunDir\best.pt" `
    --csv data\GenVideo-Val\splits\ood_test.csv `
    --batch-size 1 `
    --amp `
    --out-dir $OutDir
}

