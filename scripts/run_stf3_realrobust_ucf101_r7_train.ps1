$ErrorActionPreference = "Stop"
Set-Location "D:\VsCode Program\Python\content_security\final_project"

.\.venv\Scripts\python.exe -m src.train `
  --data-root data `
  --train-csv data\STF3-RealRobust\splits\ood_train_genvideo_ucf101_real2000.csv `
  --val-csv data\STF3-RealRobust\splits\ood_val_genvideo_prefixed.csv `
  --model stf3_new `
  --epochs 5 `
  --batch-size 1 `
  --num-workers 0 `
  --num-frames 8 `
  --image-size 224 `
  --lr 1e-4 `
  --weight-decay 1e-4 `
  --foundation-backbone dinov2_vits14 `
  --wavelet-aug-prob 0.1 `
  --wavelet-aug-mode batch `
  --branch-dropout 0.1 `
  --d3-loss l2 `
  --aux-loss-weight 0.2 `
  --pos-weight none `
  --fake-loss-weight 1.2 `
  --amp `
  --out-dir runs\ood_stf3_new_224_fakew12_ucf101real2000
