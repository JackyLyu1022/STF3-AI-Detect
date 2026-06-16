$ErrorActionPreference = "Stop"
Set-Location "D:\VsCode Program\Python\content_security\final_project"

.\.venv\Scripts\python.exe -m src.evaluate `
  --checkpoint runs\ood_stf3_new_224_fakew12_ucf101real2000\best.pt `
  --data-root data\AEGIS `
  --csv data\AEGIS\splits\aegis_hard_test.csv `
  --batch-size 1 `
  --num-workers 0 `
  --amp `
  --out-dir outputs\aegis_stf3_new_224_fakew12_ucf101real2000
