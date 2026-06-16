# Convenience commands for STF3-Detect. Run selected blocks manually.

cd "D:\VsCode Program\Python\content_security\final_project"

# Smoke test
.\.venv\Scripts\python.exe -m src.train --model spatial --epochs 1 --batch-size 2 --num-frames 4 --image-size 112 --max-train-samples 12 --max-val-samples 8 --out-dir runs\smoke_spatial
.\.venv\Scripts\python.exe -m src.evaluate --checkpoint runs\smoke_spatial\best.pt --csv data\GenVideo-Val\splits\random_test.csv --batch-size 2 --max-samples 8 --out-dir outputs\smoke_eval
.\.venv\Scripts\python.exe -m src.visualize --predictions outputs\smoke_eval\predictions.csv --history runs\smoke_spatial\history.json --out-dir outputs\smoke_figures
