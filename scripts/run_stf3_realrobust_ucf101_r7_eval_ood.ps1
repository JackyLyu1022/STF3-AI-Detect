$ErrorActionPreference = "Stop"
Set-Location "D:\VsCode Program\Python\content_security\final_project"

.\.venv\Scripts\python.exe -m src.evaluate `
  --checkpoint runs\ood_stf3_new_224_fakew12_ucf101real2000\best.pt `
  --data-root data `
  --csv data\STF3-RealRobust\splits\ood_test_genvideo_prefixed.csv `
  --batch-size 1 `
  --num-workers 0 `
  --amp `
  --out-dir outputs\ood_stf3_new_224_fakew12_ucf101real2000
