Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "D:\VsCode Program\Python\content_security\final_project"

# Main STF3-New random split experiment.
# First run may download DINOv2 weights through torch.hub.
python -m src.train `
  --model stf3_new `
  --train-csv data\GenVideo-Val\splits\random_train.csv `
  --val-csv data\GenVideo-Val\splits\random_val.csv `
  --epochs 5 `
  --batch-size 1 `
  --num-frames 8 `
  --image-size 224 `
  --foundation-backbone dinov2_vits14 `
  --wavelet-aug-prob 0.1 `
  --branch-dropout 0.1 `
  --aux-loss-weight 0.2 `
  --amp `
  --out-dir runs\random_stf3_new

python -m src.evaluate `
  --checkpoint runs\random_stf3_new\best.pt `
  --csv data\GenVideo-Val\splits\random_test.csv `
  --batch-size 1 `
  --amp `
  --out-dir outputs\random_stf3_new

