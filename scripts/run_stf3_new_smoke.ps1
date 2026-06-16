Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "D:\VsCode Program\Python\content_security\final_project"

# Frequency-only smoke does not download DINOv2/XCLIP weights.
python -m src.train `
  --model frequency_wave `
  --epochs 1 `
  --batch-size 2 `
  --num-frames 4 `
  --image-size 112 `
  --max-train-samples 8 `
  --max-val-samples 8 `
  --wavelet-aug-prob 0.0 `
  --out-dir runs\smoke_frequency_wave

python -m src.evaluate `
  --checkpoint runs\smoke_frequency_wave\best.pt `
  --csv data\GenVideo-Val\splits_mini\mini_test.csv `
  --batch-size 2 `
  --num-frames 4 `
  --image-size 112 `
  --max-samples 8 `
  --out-dir outputs\smoke_frequency_wave

