# TI / DDA 加噪方式说明（当前版本）

本文档说明当前 `prepare_label_flip_noise.py` 的加噪逻辑，以及实际运行方法。

## 1. 适用数据与输出目录

输入数据集：

- `data_updated/my_dataset_ti`
- `data_updated/my_dataset_dda`

会生成并覆盖以下 6 个含噪目录：

- `data_updated/my_dataset_ti_noise_5pct`
- `data_updated/my_dataset_ti_noise_10pct`
- `data_updated/my_dataset_ti_noise_20pct`
- `data_updated/my_dataset_dda_noise_5pct`
- `data_updated/my_dataset_dda_noise_10pct`
- `data_updated/my_dataset_dda_noise_20pct`

## 2. 当前加噪策略（分级累计 + 强干扰）

### 2.1 翻转数量严格控制

设训练集样本数为 `N`，三档目标翻转数为：

- 5%: `round(N * 0.05)`
- 10%: `round(N * 0.10)`
- 20%: `round(N * 0.20)`

每个噪声版本实际改变的标签条数就是该目标值，不会超出。

### 2.2 累计嵌套

翻转集合是累计的：

- 10% 包含 5% 的翻转样本
- 20% 包含 10% 的翻转样本

这样可以保证噪声强度随比例稳定增加，减少随机重采样带来的抖动。

### 2.3 翻转方向

主翻转（破坏真实信号）：

- `1 -> 0`
- `2 -> 0`

辅翻转（制造假边）：

- 部分 `0 -> 非0`

其中 `0 -> 非0` 的目标类选择规则：

- 在当前训练集里比较 1 类与 2 类数量
- 翻向较少的那一类（minority class）

### 2.4 三档中 0 类翻转占比

`0 -> 非0` 在总翻转中的占比：

- 5% 档：约 20%
- 10% 档：约 30%
- 20% 档：约 40%

剩余翻转全部来自 `1/2 -> 0`。

### 2.5 关系索引稳定保护

为了避免 `relation_vocab` 出现 1/2 对调，脚本会保护训练集中标签 `0/1/2` 首次出现的三条样本，不参与翻转。

## 3. 数据切分保持原则

只修改训练集：

- `train.txt`：加噪
- `valid.txt`：原样复制
- `test.txt`：原样复制

## 4. 脚本位置与关键参数

脚本文件：

- `prepare_label_flip_noise.py`

关键参数：

- `NOISE_RATIOS = (0.05, 0.10, 0.20)`
- `ZERO_FRACTION_BY_RATIO = {0.05: 0.20, 0.10: 0.30, 0.20: 0.40}`
- `BASE_SEED = 20260713`

## 5. 运行方法

在 `exp1_accuracy/NBFNet` 目录下执行：

```bash
python prepare_label_flip_noise.py
```

运行后会打印每个数据集/比例的：

- 翻转数量
- 迁移统计（如 `1->0`, `2->0`, `0->2`）
- 加噪前后标签分布

## 6. 快速校验命令

### 6.1 校验翻转数量、嵌套关系、valid/test 不变

```bash
python - <<'PY'
from pathlib import Path
base = Path('data_updated')
for ds in ['my_dataset_ti', 'my_dataset_dda']:
    print('\n===', ds, '===')
    orig = (base / ds / 'train.txt').read_text().splitlines()
    prev = set()
    for pct in [5, 10, 20]:
        noisy_name = f'{ds}_noise_{pct}pct'
        noisy = (base / noisy_name / 'train.txt').read_text().splitlines()
        changed = {i for i, (a, b) in enumerate(zip(orig, noisy)) if a != b}
        print(f'{pct}% changed={len(changed)} nested={prev.issubset(changed)}')
        prev = changed
        vsame = (base / ds / 'valid.txt').read_text() == (base / noisy_name / 'valid.txt').read_text()
        tsame = (base / ds / 'test.txt').read_text() == (base / noisy_name / 'test.txt').read_text()
        print(' valid_same=', vsame, 'test_same=', tsame)
PY
```

### 6.2 校验 relation_vocab 顺序

```bash
python - <<'PY'
from torchdrug import core
import os, sys
sys.path.append(os.getcwd())
import nbfnet.dataset

for d in [
    'my_dataset_ti_noise_5pct','my_dataset_ti_noise_10pct','my_dataset_ti_noise_20pct',
    'my_dataset_dda_noise_5pct','my_dataset_dda_noise_10pct','my_dataset_dda_noise_20pct'
]:
    ds = core.Configurable.load_config_dict({
        'class': 'MyCustomDataset',
        'path': f'/root/autodl-tmp/exp1_accuracy/NBFNet/data_updated/{d}',
        'verbose': 0,
    })
    print(d, ds.relation_vocab)
PY
```

通常应为：`['0', '1', '2']`。

## 7. 覆盖前备份建议

脚本会覆盖现有噪声目录，建议先备份：

```bash
backup_root="data_updated_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_root"
for d in my_dataset_ti_noise_5pct my_dataset_ti_noise_10pct my_dataset_ti_noise_20pct \
         my_dataset_dda_noise_5pct my_dataset_dda_noise_10pct my_dataset_dda_noise_20pct; do
  cp -a "data_updated/$d" "$backup_root/"
done
echo "$backup_root"
```
