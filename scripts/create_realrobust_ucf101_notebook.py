from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    nb_path = Path("notebooks/STF3_RealRobust_UCF101_Experiment.ipynb")
    nb_path.parent.mkdir(parents=True, exist_ok=True)

    cells: list[dict] = []

    def md(src: str) -> None:
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": src.strip("\n").splitlines(True),
            }
        )

    def code(src: str) -> None:
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": src.strip("\n").splitlines(True),
            }
        )

    md(
        """
# STF3-RealRobust UCF101 实验 Notebook

目标：在原 STF3-New R7_224_FAKEW12 基础上，加入 UCF-101 真实视频样本重新训练，观察是否能降低 AEGIS 外部测试集上的真实视频误报（FP）。

本 notebook 只做实验调度：

- 不在 notebook 中重写训练逻辑；
- 训练仍调用 `src.train`；
- 测试仍调用 `src.evaluate`；
- 数据集 CSV 由 `scripts/build_realrobust_ucf101_splits.py` 生成；
- 建议一次只运行一个实验单元，不建议直接 Run All。
"""
    )

    md(
        """
## 0. 实验说明

本实验使用：

```text
Train: GenVideo OOD train + UCF-101 real 2000
Val:   GenVideo OOD val
Test1: GenVideo OOD test
Test2: AEGIS hard test set
```

重要原则：

> AEGIS 不参与训练或验证，只作为外部测试集。
"""
    )

    code(
        """
from pathlib import Path
import json
import csv
from collections import Counter

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == 'notebooks':
    PROJECT_ROOT = PROJECT_ROOT.parent

print('PROJECT_ROOT =', PROJECT_ROOT)
assert (PROJECT_ROOT / 'src' / 'train.py').exists(), '请从 final_project 根目录或 notebooks 中启动 Jupyter'
"""
    )

    md(
        """
## 单行进度运行器

下面这个函数用于在 Jupyter 中压缩显示训练/测试进度：

- 只保留一个动态输出块；
- 默认每 10 秒刷新一次；
- 显示已运行时间、最后一条日志和最近几条日志；
- 不设置 `TQDM_*` 环境变量，避免之前 tqdm 在本环境中的兼容问题。
"""
    )

    code(
        """
from IPython.display import clear_output
import subprocess, time, os, re


def _shorten(text, max_len=220):
    text = str(text).replace('\\r', ' ').replace('\\n', ' ')
    text = re.sub(r'\\s+', ' ', text).strip()
    if len(text) > max_len:
        return text[-max_len:]
    return text


def run_powershell_compact(script_path, refresh_seconds=10, tail_lines=8):
    \"\"\"在 Jupyter 中以单块动态输出运行 PowerShell 脚本。\"\"\"
    script_path = Path(script_path)
    assert script_path.exists(), f'脚本不存在: {script_path}'

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'

    ps_command = (
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); "
        "$OutputEncoding = [System.Text.UTF8Encoding]::new(); "
        f"& '{script_path}'"
    )
    cmd = ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', ps_command]
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
        env=env,
    )

    recent = []
    last_line = ''
    last_refresh = 0.0

    try:
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if line:
                parts = re.split(r'[\\r\\n]+', line)
                parts = [x for x in parts if x.strip()]
                if parts:
                    last_line = parts[-1]
                    recent.extend(parts)
                    recent = recent[-tail_lines:]

            now = time.time()
            if now - last_refresh >= refresh_seconds:
                clear_output(wait=True)
                elapsed = now - start
                print(f'Running: {script_path.name}')
                print(f'Elapsed: {elapsed/60:.1f} min ({elapsed:.0f} s)')
                print(f'Last: {_shorten(last_line)}')
                print('\\nRecent logs:')
                for x in recent[-tail_lines:]:
                    print('  ' + _shorten(x, 180))
                last_refresh = now

            if line == '' and proc.poll() is not None:
                break
            if not line:
                time.sleep(0.2)

        code = proc.wait()
        clear_output(wait=True)
        elapsed = time.time() - start
        print(f'Finished: {script_path.name}')
        print(f'Exit code: {code}')
        print(f'Total time: {elapsed/60:.1f} min ({elapsed:.0f} s)')
        print('\\nFinal recent logs:')
        for x in recent[-tail_lines:]:
            print('  ' + _shorten(x, 180))
        if code != 0:
            raise subprocess.CalledProcessError(code, cmd)
        return code
    except KeyboardInterrupt:
        proc.terminate()
        raise
"""
    )

    md(
        """
## 1. 检查 RealRobust 数据集 CSV

如果这些文件不存在，请先运行数据集构建单元。
"""
    )

    code(
        """
SPLIT_DIR = PROJECT_ROOT / 'data' / 'STF3-RealRobust' / 'splits'
TRAIN_CSV = SPLIT_DIR / 'ood_train_genvideo_ucf101_real2000.csv'
VAL_CSV = SPLIT_DIR / 'ood_val_genvideo_prefixed.csv'
TEST_CSV = SPLIT_DIR / 'ood_test_genvideo_prefixed.csv'
AEGIS_CSV = PROJECT_ROOT / 'data' / 'AEGIS' / 'splits' / 'aegis_hard_test.csv'

for p in [TRAIN_CSV, VAL_CSV, TEST_CSV, AEGIS_CSV]:
    print(p.relative_to(PROJECT_ROOT), 'exists =', p.exists())
"""
    )

    md(
        """
## 2. 如有需要，重新生成 RealRobust split

默认从 UCF-101 中均匀抽样 2000 个真实视频加入训练集。

如果已经生成过，可以跳过此单元。
"""
    )

    code(
        """
# 可选：重新生成 split
# 注意：这不会覆盖原始 GenVideo split，只会写入 data/STF3-RealRobust/splits

import subprocess, sys

cmd = [
    sys.executable,
    str(PROJECT_ROOT / 'scripts' / 'build_realrobust_ucf101_splits.py'),
    '--extra-real', '2000',
    '--seed', '42',
]
print(' '.join(map(str, cmd)))
# subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
print('默认不自动执行。如需重建，请取消上面 subprocess.run 的注释后运行本单元。')
"""
    )

    md(
        """
## 3. 数据集统计

确认训练集真实视频数量已经增加，且 val/test 仍然保持 GenVideo OOD 原始划分。
"""
    )

    code(
        """
def load_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def summarize_csv(path):
    rows = load_rows(path)
    return {
        'path': str(path.relative_to(PROJECT_ROOT)),
        'n': len(rows),
        'label_name': dict(Counter(r.get('label_name', '') for r in rows)),
        'label': dict(Counter(r.get('label', '') for r in rows)),
        'generator_top10': Counter(r.get('generator', '') for r in rows).most_common(10),
    }

for p in [TRAIN_CSV, VAL_CSV, TEST_CSV, AEGIS_CSV]:
    print(json.dumps(summarize_csv(p), ensure_ascii=False, indent=2))
"""
    )

    md(
        """
## 4. 训练 STF3-RealRobust

本单元会调用：

```text
scripts/run_stf3_realrobust_ucf101_r7_train.ps1
```

输出目录：

```text
runs/ood_stf3_new_224_fakew12_ucf101real2000
```

建议：训练耗时较长，确认环境和 GPU 正常后再运行。
"""
    )

    code(
        """
import subprocess

train_script = PROJECT_ROOT / 'scripts' / 'run_stf3_realrobust_ucf101_r7_train.ps1'
print('Train script:', train_script)
assert train_script.exists()

# 运行训练：取消下一行注释即可开始。refresh_seconds 可改成 5 / 10 / 30。
# run_powershell_compact(train_script, refresh_seconds=10, tail_lines=8)
print('默认不自动训练。确认后取消 run_powershell_compact 这一行的注释并运行本单元。')
"""
    )

    md(
        """
## 5. GenVideo OOD 测试

训练完成后运行本单元，评估新的 RealRobust 模型在 GenVideo OOD test 上是否仍保持原有性能。
"""
    )

    code(
        """
ood_eval_script = PROJECT_ROOT / 'scripts' / 'run_stf3_realrobust_ucf101_r7_eval_ood.ps1'
print('OOD eval script:', ood_eval_script)
assert ood_eval_script.exists()

# 运行 OOD 测试：取消下一行注释即可开始。
# run_powershell_compact(ood_eval_script, refresh_seconds=10, tail_lines=8)
print('默认不自动测试。确认训练完成后取消 run_powershell_compact 这一行的注释并运行本单元。')
"""
    )

    md(
        """
## 6. AEGIS 外部测试

本单元用于观察加入 UCF-101 real 后，AEGIS 上的 FP 是否下降。
"""
    )

    code(
        """
aegis_eval_script = PROJECT_ROOT / 'scripts' / 'run_stf3_realrobust_ucf101_r7_eval_aegis.ps1'
print('AEGIS eval script:', aegis_eval_script)
assert aegis_eval_script.exists()

# 运行 AEGIS 测试：取消下一行注释即可开始。
# run_powershell_compact(aegis_eval_script, refresh_seconds=10, tail_lines=8)
print('默认不自动测试。确认训练完成后取消 run_powershell_compact 这一行的注释并运行本单元。')
"""
    )

    md(
        """
## 7. 读取测试结果

训练和测试完成后，运行本单元读取 `metrics.json`。
"""
    )

    code(
        """
OOD_METRICS = PROJECT_ROOT / 'outputs' / 'ood_stf3_new_224_fakew12_ucf101real2000' / 'metrics.json'
AEGIS_METRICS = PROJECT_ROOT / 'outputs' / 'aegis_stf3_new_224_fakew12_ucf101real2000' / 'metrics.json'

for name, p in [('OOD', OOD_METRICS), ('AEGIS', AEGIS_METRICS)]:
    print('\\n==', name, '==')
    print('path:', p.relative_to(PROJECT_ROOT))
    if p.exists():
        data = json.loads(p.read_text(encoding='utf-8'))
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print('尚未生成，请先运行对应评估单元。')
"""
    )

    md(
        """
## 8. AEGIS 误报重点分析

当前最关心的是 AEGIS 上的：

```text
FP 是否下降
Precision 是否上升
Recall 是否仍然可接受
```

下面单元会从混淆矩阵中提取 TN / FP / FN / TP，并计算 Specificity。
"""
    )

    code(
        """
def summarize_binary_metrics(metrics):
    cm = metrics.get('confusion_matrix')
    out = dict(metrics)
    if cm and len(cm) == 2:
        tn, fp = cm[0]
        fn, tp = cm[1]
        specificity = tn / (tn + fp) if (tn + fp) else None
        out.update({'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'specificity': specificity})
    return out

if AEGIS_METRICS.exists():
    m = summarize_binary_metrics(json.loads(AEGIS_METRICS.read_text(encoding='utf-8')))
    keys = ['acc', 'auc', 'f1', 'precision', 'recall', 'specificity', 'tn', 'fp', 'fn', 'tp']
    for k in keys:
        print(f'{k}: {m.get(k)}')
else:
    print('AEGIS metrics 尚未生成。')
"""
    )

    md(
        """
## 9. 实验记录模板

跑完后可以把结果复制到：

```text
docs/AEGIS数据集测试汇总.md
docs/STF3_RealRobust_UCF101实验方案.md
```

建议记录：

- OOD test 指标是否下降；
- AEGIS FP 是否下降；
- AEGIS Precision 是否上升；
- AEGIS Recall 是否过度下降；
- 是否值得继续增加 UCF-101 real 数量或换 WebVid/Pexels real。
"""
    )

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": ".venv final_project (Python 3.12.7)",
                "language": "python",
                "name": "final_project_venv",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.7",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[write] {nb_path} cells={len(cells)}")


if __name__ == "__main__":
    main()
