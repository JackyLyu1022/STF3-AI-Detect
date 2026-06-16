$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$ProjectRoot = Split-Path -Parent $ScriptDir
$Py = Join-Path $ScriptDir ".venv_d3\Scripts\python.exe"
$Weights = Join-Path $ScriptDir "WaveRep\demo\weights\weights_dinov2_G4.ckpt"

if (-not (Test-Path $Py)) {
    throw "D3/WaveRep shared venv not found: $Py"
}

if (-not (Test-Path (Join-Path $ScriptDir "WaveRep"))) {
    throw "WaveRep repo not found under comparison_experiment\WaveRep"
}

if (-not (Test-Path $Weights)) {
    throw @"
WaveRep G4 weights are not found:
  $Weights

Download command:
  Invoke-WebRequest -Uri "https://www.grip.unina.it/download/prog/WaveRep_SynthVideoDet/weights_dinov2_G4.ckpt" -OutFile "$Weights" -UseBasicParsing
"@
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

& $Py (Join-Path $ScriptDir "run_waverep_ood_compare.py") `
  --project-root $ProjectRoot `
  --waverep-root "comparison_experiment/WaveRep" `
  --data-root "data/GenVideo-Val" `
  --val-csv "data/GenVideo-Val/splits/ood_val.csv" `
  --test-csv "data/GenVideo-Val/splits/ood_test.csv" `
  --out-dir "comparison_experiment/results/WaveRep/ood_g4_uniform8" `
  --weights "comparison_experiment/WaveRep/demo/weights/weights_dinov2_G4.ckpt" `
  --device "cuda:0" `
  --num-frames 8 `
  --crop-size 504 `
  --batch-size 8
