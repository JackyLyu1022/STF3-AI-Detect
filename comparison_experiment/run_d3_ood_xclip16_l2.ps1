$ErrorActionPreference = "Stop"

# Run from project root:
#   D:\VsCode Program\Python\content_security\final_project
#
# D3 is training-free. This script:
# 1) scores OOD val/test videos with D3 XCLIP-16 + L2 second-order feature;
# 2) chooses score orientation and thresholds on OOD val only;
# 3) evaluates fixed thresholds on OOD test.

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot "comparison_experiment\.venv_d3\Scripts\python.exe"

& $Python "comparison_experiment\run_d3_ood_compare.py" `
  --encoder XCLIP-16 `
  --loss l2 `
  --device cuda:0 `
  --num-frames 16 `
  --image-size 224 `
  --out-dir "comparison_experiment\results\D3\ood_xclip16_l2"

