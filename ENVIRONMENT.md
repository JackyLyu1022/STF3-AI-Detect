# Final Project Environment

本项目本地虚拟环境位于：`.venv/`

## 激活

PowerShell：

```powershell
cd "D:\VsCode Program\Python\content_security\final_project"
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 执行策略不允许激活脚本，可使用：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

或不激活，直接运行：

```powershell
.\.venv\Scripts\python.exe scripts\check_gpu.py
```

## GPU 检查

```powershell
python scripts\check_gpu.py
nvidia-smi
```

当前已安装 PyTorch CUDA 版本：`torch==2.11.0+cu128`，已验证 RTX 4060 Laptop GPU 可用。

## 数据目录建议

不要把大数据集提交到代码仓库；建议后续使用：

```text
data/
  GenVideo-Val/
  GenVideo-100K/
  GenVidBench-143K/
checkpoints/
runs/
```

这些目录应加入 `.gitignore`。
