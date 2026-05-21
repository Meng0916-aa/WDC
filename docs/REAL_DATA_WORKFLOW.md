# Real .xtherm Data Workflow — Detailed Walkthrough

> 配套于 `README.md` 的"真实数据接入工作流"小节；本文件展开每一步的预期输出、典型故障与排查思路。
> 与项目约定（`CLAUDE.md`）保持一致：
> - 原始 `.xtherm` **不入库**，数据库只保存路径与元数据；
> - 不存在 synthetic experiment pipeline；
> - 当前阶段不写 PyTorch / 深度学习模型。

---

## 0. 前置条件

- 项目根目录：`D:\GEJ-WDC`，所有命令都在根目录下运行。
- 已完成 `pip install -r requirements.txt`。
- 真实 `.xtherm` 永远只放在 `data/raw/<group>/<sample>/` 下（建议按磁场强度分组，如 `data/raw/B040/sample01/`）。
- 一台 Xiris VXIR-3000 在固定工作距离上拍摄；如果换工作距离或镜头，`dx/dy` 标定必须重做。

---

## 1. 预检查 (`scripts/check_ready_for_real_data.py`)

```bash
python scripts/check_ready_for_real_data.py --config configs/default.yaml
```

**预期输出**（项目刚初始化、尚未标定 dx/dy 时）：

```
Project readiness preflight (D:\GEJ-WDC\configs\default.yaml)
=============================================================
[ OK ] database                  : D:\GEJ-WDC\database\thermal_cladding.db
[INFO] xtherm_files registered   : 0
[ OK ] data/raw                  : D:\GEJ-WDC\data\raw
...
[WARN] camera.dx_mm_per_pixel    : null (camera calibration not yet measured)
[WARN] camera.dy_mm_per_pixel    : null (camera calibration not yet measured)
[WARN] gradient feasibility      : without dx/dy, ...
[INFO] camera.header_offset      : 0 (default; ...)

Summary
-------
ready_for_registration        : YES
ready_for_probing             : YES
ready_for_gradient_processing : NO
```

**退出码**

| code | 含义 | 应该做什么 |
| ---: | --- | --- |
| `0` | 全绿，可登记 + 探测 + 计算梯度 + 批处理 | 继续 Step 8 onwards |
| `1` | 可登记可探测，**不能算 degC/mm 温度梯度**（dx/dy 未填） | 可登记 + 探测，但**不要**跑批处理；先做相机标定 |
| `2` | 基础结构不完整（数据库 / 目录 / 配置缺失） | 先跑 `init_database.py`，或修复 YAML / 目录 |

CI / Makefile 用法：

```bash
python scripts/check_ready_for_real_data.py && python scripts/batch_process_database.py --status registered
```

`&&` 会在退出码非 0 时拦住后续命令，避免一批文件全部被记 `error`。

---

## 2. 初始化数据库 (`init_database.py`)

```bash
python scripts/init_database.py --config configs/default.yaml
```

幂等：重跑不会删数据。建表后会写入一行 `experiments`（按 `configs/default.yaml → experiment.name` 命名）。

---

## 3. 登记真实 `.xtherm` 文件 (`register_xtherm_files.py`)

```bash
# 单文件
python scripts/register_xtherm_files.py --config configs/default.yaml \
    --input data/raw/B000/sample01/run01.xtherm \
    --sample-id B000_sample01 --B-mT 0

# 递归扫描整个 sample 目录
python scripts/register_xtherm_files.py --config configs/default.yaml \
    --input data/raw/B080/sample02 \
    --sample-id B080_sample02 --B-mT 80 --recursive

# 计算 SHA-256（用作完整性校验，大文件会变慢）
python scripts/register_xtherm_files.py --config configs/default.yaml \
    --input data/raw/B120 --sample-id B120_sample01 --B-mT 120 --recursive --sha256
```

**注意点**：

- `--sample-id` 是逻辑标识，**必须唯一**；同名但 `--B-mT` 不一致会被拒绝（防止串样本）。
- 同一文件路径重复登记不会报错，会返回已有 `file_id`，方便脚本反复跑。
- 登记时 `status='registered'`、`header_offset` 与 `(W,H)` 取自 `configs/default.yaml`，**这两个值通常需要在 Step 4 经 probe 校核后才确认**。

---

## 4. 格式探测 (`probe_registered_file.py`)

```bash
python scripts/probe_registered_file.py --config configs/default.yaml --file-id 1
```

**预期输出**（示意）：

```
  offset    n_frames     Tmin       Tmax     Tmean   note
       0       1234     20.50    1820.30    250.10
     128       1233     20.50    1820.30    250.10
     256          0         -          -         -   not divisible (ValueError)
     512          0         -          -         -   not divisible (ValueError)
    1024          0         -          -         -   not divisible (ValueError)
    2048          0         -          -         -   not divisible (ValueError)
    4096          0         -          -         -   not divisible (ValueError)
  Best  : offset=0 n_frames=1234
```

**应该看什么**

1. **`n_frames` 合不合理** —— 用 `采集时长 (s) × fps` 估算。例：5 秒 @ 180 fps ≈ 900 帧。
2. **`Tmin / Tmax` 是否物理上合理** —— 基板预热温度（室温 20°C 起）+ 熔池峰值温度（CoCrNi 体系约 1400–1800°C）。
3. **是否多个 offset 都能整除** —— 若 `0` 和 `128` 都给出相近 `n_frames`（差 1 是因为 128 字节 < 一帧字节数），优先选 `0`。

满意后写回数据库：

```bash
python scripts/probe_registered_file.py --config configs/default.yaml --file-id 1 --apply
```

`--apply` 会更新该 `file_id` 的 `header_offset`、`estimated_frames`，并把 `status` 推到 `probed`。

> 如果没有任何 offset 候选给出合理温度范围，**不要 `--apply`**。先检查 Xiris 导出设置（是否真的是 16-bit raw、scale 是否 0.1、是否启用了 AOI / 1000 fps 模式）；必要时手动改 `configs/default.yaml` 里的 `width / height / dtype / temperature_scale`，重新登记再探测。

---

## 5. 相机空间标定（dx / dy）

**推荐做法**（在拍摄完真实数据后做，相机姿态不能再动）：

1. 拍一张已知尺寸的标定靶 / 已知尺寸的工件 / 已知道间距的搭接面；
2. 在 Xiris WeldStudio 或自写的 Python 脚本里量出靶标在像素上的跨度（例：靶标实际 `10 mm`，像素跨度 `200 px`，则 `dx_mm_per_pixel = 10 / 200 = 0.05`）；
3. 默认相机像素是正方形，`dx ≈ dy`；若投影面与相机光轴不正交（斜射），`dx` 与 `dy` 不一定相等，分别量。
4. 把数值写入 `configs/default.yaml`：

```yaml
camera:
  ...
  dx_mm_per_pixel: 0.05
  dy_mm_per_pixel: 0.05
```

5. 再跑一次预检查，确认 `exit_code=0`：

```bash
python scripts/check_ready_for_real_data.py --config configs/default.yaml
```

---

## 6. 单文件处理 (`process_registered_file.py`)

```bash
python scripts/process_registered_file.py --config configs/default.yaml --file-id 1
```

**做了什么**

- 读取 `.xtherm` → float32 `[T,H,W]` 摄氏度 cube
- 逐帧计算 `Gx, Gy, G`（单位 `degC/mm`）
- 逐帧提取 `Tmax / Tmean / Tstd / Gmax / Gmean / Gstd / high_temp_area`
- 写入 `frame_features` 表（每帧一行）
- 写入 `processing_results` 摘要（每文件一行：`Tmax_global / Tmean_global / Gmax_global / Gmean_global / n_frames`）
- 落特征 CSV 到 `data/features/file<id>__sample<pk>__<basename>.csv`
- `status` 推到 `processed`

**先跑一个文件**，目测 CSV：

- 每帧 `Tmax` 走势应该看到加热—峰值—冷却的脉冲式分布（每过一个激光点就一个峰）；
- `high_temp_area` 在峰值帧附近不应该为 0（熔池存在）；
- `Gmax` 数量级合理：熔池边界梯度通常 `O(10^2–10^3) degC/mm`。

确认无误后再进入 Step 7 批处理。

---

## 7. 批处理 (`batch_process_database.py`)

```bash
# 默认: 处理所有 status='registered'
python scripts/batch_process_database.py --config configs/default.yaml --status registered

# 已 probe 过的:
python scripts/batch_process_database.py --config configs/default.yaml --status probed

# 显式 id 列表 (会忽略 --status)
python scripts/batch_process_database.py --config configs/default.yaml --file-ids 1,3,5,7

# 全量重跑 (重跑安全, 内部 INSERT OR REPLACE)
python scripts/batch_process_database.py --config configs/default.yaml --status all

# 首次失败即停 (调参 / 调试时建议加)
python scripts/batch_process_database.py --config configs/default.yaml --status registered --stop-on-error
```

退出码：`0`=全部成功；`2`=有失败（详情见 `processing_results.error_message`）。

---

## 8. 导出汇总特征 (`export_features_from_database.py`)

```bash
# 全量
python scripts/export_features_from_database.py --config configs/default.yaml
# 默认输出: data/features/all_features.csv

# 按磁场分组
python scripts/export_features_from_database.py --config configs/default.yaml --B-mT 80

# 按 sample
python scripts/export_features_from_database.py --config configs/default.yaml --sample-id B000_sample01

# 按 file_id 列表
python scripts/export_features_from_database.py --config configs/default.yaml --file-ids 1,2,3 \
    --output data/features/subset_B0.csv
```

输出每行带 `sample_id / B_mT / laser_power_W / scan_speed_mm_per_min / powder_feed_rate_g_per_min / hatch_spacing_mm`，可直接喂入论文图表脚本。

---

## 9. 测试

```bash
python -m pytest tests/ -q
```

测试只用 288 字节的临时二进制 fixture（`W=8, H=6, T=3, uint16, scale=0.1`），**不依赖任何真实实验数据**。这不构成 synthetic experiment pipeline——它只是为了让 reader / probe / gradient / features / db 五个模块能在没有相机的环境下回归。

---

## 相机模式 / 配置变更对照

| 场景 | 需要改 `configs/default.yaml` 哪里 | 备注 |
| --- | --- | --- |
| 启用 AOI（裁切视场） | `camera.width`, `camera.height` | 跟 Xiris WeldStudio 里设置的 AOI 一致 |
| 1000 fps 高速模式 | `camera.fps=1000`, 并核对 `width/height`（很可能也被裁了） | `time_s` 才能算对 |
| 帧率不是 180 fps | `camera.fps=<真实值>` | 否则 `frame_features.time_s` 偏 |
| 16-bit raw 不是 ×0.1 | `camera.temperature_scale=<真实值>` | 视 Xiris 导出配置 |
| 换相机或换工作距离 | 重新标定 `dx_mm_per_pixel / dy_mm_per_pixel` | 上一组标定值作废 |

---

## 从 `status='error'` 恢复

1. 查异常摘要：

   ```sql
   SELECT xf.id, xf.file_path, pr.error_message
   FROM xtherm_files xf
   JOIN processing_results pr ON pr.xtherm_file_id = xf.id
   WHERE xf.status = 'error'
   ORDER BY xf.id;
   ```

2. 根据信息修源头：
   - `dx_mm_per_pixel and dy_mm_per_pixel are required` → 标定后改 YAML，再 `--status error` 重跑。
   - `silent reshape disabled` → 形状或 offset 不对，重跑 probe，必要时改 `width/height/dtype`。
   - `xtherm file not found` → 文件被移动 / 删除，恢复路径或重新登记。

3. 修完重跑（pipeline 用 `INSERT OR REPLACE`，特征会覆盖更新）：

   ```bash
   python scripts/batch_process_database.py --config configs/default.yaml --status error
   ```

---

## 单帧文件夹合并 (One-frame-per-file → Sequence NPZ)

Xiris WeldStudio 有两种 `.xtherm` 导出风格：

| 导出方式 | 一个文件包含 | 处理路径 |
| --- | --- | --- |
| **多帧合一** | 整段录制的所有帧 (header + N × frame_bytes) | 走 register + probe + process 主流程 |
| **一帧一文件** | 一帧 (header + 1 × frame_bytes) | **先 merge，再 dedup，再 process** |

判别方式：probe 第一个文件后，如果 `n_frames = 1` 且文件夹里有大量同形态文件，几乎可以确定是"一帧一文件"模式。本项目的当前实际数据就是这种：

- 文件大小 = **655416 bytes** (`640 × 512 × uint16 = 655360` + **header_offset = 56**)
- `n_frames = 1`, Tmin / Tmax / Tmean 物理合理

### 合并步骤

```bash
python scripts/merge_xtherm_folder.py --config configs/default.yaml \
    --input-dir data/raw/B000/sample01 \
    --sample-id B000_sample01 --B-mT 0 \
    --output data/processed/B000_sample01_temperature_sequence.npz
```

可选参数：

- `--pattern "*.xtherm"`：默认匹配，需要兼容大小写或其它扩展名时改这里。
- `--recursive`：扫描 `--input-dir` 下的所有子目录。

### 关键行为

1. **不修改原始 `.xtherm` 文件**——纯只读读取。
2. **按文件名自然排序**：`frame2.xtherm` 排在 `frame10.xtherm` 之前（普通字典序会反过来）。
3. **每个文件严格按 1 帧验证**：用 `read_xtherm(..., expected_frames=1)`；尺寸对不上直接抛错，不做 silent reshape。
4. **温度统一换算**：`raw_value * temperature_scale = degC`；输出 `float32`，shape `[T, H, W]`。

### 输出 NPZ 字段

必填（spec 要求）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `temperature` | float32 `[T,H,W]` | 摄氏度温度序列 |
| `source_files` | str array | 合并顺序对应的相对路径（PROJECT_ROOT 之下时） |
| `fps` | float | 来自 `configs/default.yaml` |
| `width`, `height` | int | 来自 config |
| `header_offset` | int | 来自 config（项目实际是 56） |
| `temperature_scale` | float | 来自 config（项目默认 0.1） |
| `sample_id` | str | 来自 CLI |
| `B_mT` | float | 来自 CLI |

附加溯源字段（与 `xtherm-reader / experiment-metadata` skills 约定一致，方便下游 dedup / process / export 直接用）：

`dtype`, `endian`, `run_id`, `laser_power_w`, `scan_speed_mm_min`, `powder_feed_g_min`, `powder`, `substrate`, `n_frames`, `input_dir`。

### 衔接重复帧检测

合并完成的 NPZ 可以直接喂给下一段 QA：

```bash
python scripts/detect_duplicate_frames.py --config configs/default.yaml \
    --npz data/processed/B000_sample01_temperature_sequence.npz
```

`detect_duplicate_frames.py` 的 `--npz` 模式只要求输入 NPZ 含 `temperature` 键；其他溯源字段（`fps`, `sample_id`, `B_mT`, ...）如果存在，会被自动透传到 dedup 输出的 NPZ 中，避免链路里手动复制配置。

### 工作流位置

```
真实 xtherm 抵达
  ├─ 多帧合一文件: register → probe → calibrate → check_ready → process → export
  └─ 一帧一文件 : merge_xtherm_folder → detect_duplicate_frames
                   → extract_features_from_npz → 后续绘图 / PyTorch
```

第二条路径目前不强制写回数据库——`scripts/extract_features_from_npz.py` 直接产出 features CSV，可立即喂入论文图表脚本 / PyTorch dataset；若仍希望以数据库为唯一事实源，可把合并后的单个大文件再次登记进 `xtherm_files`（其时 `n_frames` 等于合并后的帧数）。

---

## 重复帧检测与去重 (Duplicate-frame QA)

在 Xiris VXIR-3000 + WeldStudio 高负载或高速 AOI 模式下，偶尔会把同一帧 raw payload 流式输出两次。这会让帧数虚高、`Tmean` 偏倚、`time_s = frame_index / fps` 时间轴整体右移。本工具是一道**独立 QA 阶段**——

**核心原则**

1. **不要直接修改或删除原始 `.xtherm` 文件**——它是溯源链的根。
2. 推荐路线：生成一个**去重后的 NPZ + 一份审计 CSV**，原 `.xtherm` 保持只读。
3. **exact duplicate** (`np.array_equal` 相邻两帧完全一致) 默认会被自动剔除。
4. **near duplicate** (差异 ≤ MAE / max_abs 阈值) 默认**只标记，不删除**——边界态留给人工判断。
5. NPZ 中保留 `original_frame_indices`，可在后续重建任何按帧索引/时间的对齐关系。
6. 这一步**不会改 `xtherm_files.status`，也不会覆盖 `processing_results`**——纯旁路工具。

### 调用方式

两种输入方式互斥：

```bash
# A. 从数据库登记记录读取 (推荐): 自动带 sample_id / B_mT / run_id 等元数据
python scripts/detect_duplicate_frames.py --config configs/default.yaml --file-id 1

# 严格一些, 同时检测近似重复 (只标记)
python scripts/detect_duplicate_frames.py --config configs/default.yaml --file-id 1 \
    --mae-threshold 0.5 --max-abs-threshold 5.0

# 真正要把 near duplicate 也删掉时, 加 --remove-near-duplicates
python scripts/detect_duplicate_frames.py --config configs/default.yaml --file-id 1 \
    --mae-threshold 0.5 --max-abs-threshold 5.0 --remove-near-duplicates

# B. 直接对一个已有 NPZ 做去重 (该 NPZ 必须含 key 'temperature', shape [T,H,W])
python scripts/detect_duplicate_frames.py --config configs/default.yaml \
    --npz data/processed/B000_sample01_temperature.npz
```

阈值单位与输入数组单位相同。pipeline 输出的温度是 `degC`，所以 `--mae-threshold 0.5` 表示"逐像素平均偏差 ≤ 0.5 degC 视为近似重复"。

### 输出

| 文件 | 路径 (默认) | 内容 |
| --- | --- | --- |
| 审计 CSV | `data/features/{sample_id}_file{file_id}_duplicate_frames.csv` | 每帧一行：`frame_index, prev_frame_index, is_exact_duplicate, mae_to_prev, max_abs_diff_to_prev, is_near_duplicate, action` |
| 去重 NPZ | `data/processed/{sample_id}_file{file_id}_temperature_dedup.npz` | 见下 |

NPZ 内嵌的字段（用 `np.load` 后按 key 取值）：

- **温度数据**：`temperature` (shape `[T_kept, H, W]`, float32 degC), `width`, `height`
- **重建时间轴**：`original_frame_indices` (留作回归原始时间), `removed_frame_indices`, `fps`
- **溯源**：`source_file_id` 或 `source_npz`（互斥）, `source_file`(相对路径)
- **实验元数据**（仅 `--file-id` 模式带）：`sample_id`, `B_mT`, `run_id`, `laser_power_w`, `scan_speed_mm_min`, `powder_feed_g_min`, `powder`, `substrate`, `temperature_scale`, `dtype`, `endian`, `header_offset`
- **审计指针**：`duplicate_report_path`（指向 CSV 的相对路径）
- **统计**：`n_frames_original`, `n_frames_kept`, `n_frames_removed`, `n_exact_duplicates`, `n_near_duplicates`, `remove_near_duplicates`, `mae_threshold`, `max_abs_threshold`

### 重建原始时间轴

如果后续要在论文 / 模型里恢复"被剔除的帧应当在哪个时间点"，用：

```python
import numpy as np
data = np.load("data/processed/B000_sample01_file1_temperature_dedup.npz")
original_indices = data["original_frame_indices"]     # 每个保留帧的原始 idx
fps = float(data["fps"])
time_s = original_indices.astype(np.float64) / fps    # 真实时间, 不是 0..T_kept-1 / fps
```

直接用 `np.arange(T_kept) / fps` 会让时间轴漂移；务必用 `original_frame_indices`。

### 何时跑这一步

- **建议**：把它放在 `register → probe → (calibrate dx/dy) → check_ready=0 → 单文件处理 → 批处理` 流程的**单文件处理之前**——一旦发现 exact duplicate 比例显著，先去重再算梯度 / 特征，可避免误差累积。
- **可选**：所有文件都跑过批处理后再回头跑一次 dedup，对比 `data/features/all_features.csv` 与 dedup 后的特征，作为质控复核。

### 典型阈值起点

| 工况 | `--mae-threshold` | `--max-abs-threshold` |
| --- | --- | --- |
| 高速 AOI / 1000 fps, 噪声相对小 | `0.1 degC` | `1.0 degC` |
| 全画幅 180 fps, 普通工况 | `0.5 degC` | `5.0 degC` |
| 强磁场试验 (B=120 mT) 噪声偏大 | `1.0 degC` | `10.0 degC` |

不确定时**先不带阈值**只跑 exact，看到底有没有真正的完全重复；有再用阈值二次扫描。

---

## 从合并 NPZ 提取特征 (Extract features from merged NPZ)

`scripts/extract_features_from_npz.py` 是 NPZ 驱动的特征提取入口——和数据库驱动的 `src.pipeline.process_run` 并列，复用同一套 `compute_gradients / extract_frame_features` 核函数，只是不读 `xtherm_files` 表、不写 `frame_features / processing_results`。

### 调用方式

```bash
# 最常用 (--output 缺省时, 自动剥离 _temperature_dedup / _temperature_sequence 后缀)
python scripts/extract_features_from_npz.py --config configs/default.yaml \
    --input data/processed/B000_sample01_temperature_sequence_temperature_dedup.npz

# 显式指定输出
python scripts/extract_features_from_npz.py --config configs/default.yaml \
    --input data/processed/B000_sample01_temperature_sequence_temperature_dedup.npz \
    --output data/features/B000_sample01_temperature_sequence_features.csv

# 额外保存一份逐帧 G 统计 NPZ (Gmax/Gmean/Gstd as arrays, 方便绘图脚本读)
python scripts/extract_features_from_npz.py --config configs/default.yaml \
    --input data/processed/B000_sample01_temperature_sequence_temperature_dedup.npz \
    --save-gradient-stats-npz data/features/B000_sample01_gradient_stats.npz
```

### 计算了什么

逐帧（默认 `gaussian_sigma_px=0.0`，可在 YAML 中调高做降噪）：

- `Gx = ∂T/∂x`、`Gy = ∂T/∂y`、`G = sqrt(Gx² + Gy²)`，单位 `degC/mm`。
- 帧特征：`Tmax / Tmean / Tstd / Gmax / Gmean / Gstd / high_temp_area`。

CSV 列顺序（用户合约）：

```
frame, time_s, Tmax, Tmean, Tstd, Gmax, Gmean, Gstd, high_temp_area
```

### time_s 优先级

| 顺位 | 来源 | 说明 |
| --- | --- | --- |
| 1 | NPZ 中 `original_frame_indices` ÷ `fps` | dedup 输出会带这个字段，**保留原始时间轴**（剔除帧后仍正确） |
| 2 | `arange(T) / fps` | NPZ 没有 `original_frame_indices` 时回退 |
| 3 | `NaN` | NPZ 与 config 都没有 fps 时（不可达，因为 config 必填 fps） |

fps 来源优先级：**NPZ.fps > config.camera.fps**。这让一段 60 fps 录制 NPZ 不会被 config 中的默认 180 fps 强行覆盖。

### 缺标定时的行为

如果 `configs/default.yaml` 里 `dx_mm_per_pixel` 或 `dy_mm_per_pixel` 是 `null`：脚本**立即抛 `ValueError`**，提示先跑 `check_ready_for_real_data.py`，不会处理半个文件就崩。

### 关于 Gx / Gy / G 三维数组

**默认不会保存完整的 `[T, H, W]` 梯度立方体**——一段 410 帧、640×512 的录制 3 个梯度通道是 `410 × 640 × 512 × 3 × 4 byte ≈ 1.6 GB`，对一段实验来说不划算。

需要时用 `--save-gradient-stats-npz <path>`，只会写入逐帧标量 `(Gmax, Gmean, Gstd)`（与 CSV 等价但 numpy 数组形式）。若以后真的需要全场梯度做可视化或 PyTorch dataset，建议按需重算或新增一个独立的"梯度立方体导出"脚本。

### 与数据库驱动路径的并列关系

| 入口 | 适用场景 | 写回数据库 |
| --- | --- | --- |
| `process_registered_file.py` / `batch_process_database.py` | 多帧合一 `.xtherm`，已登记入库 | ✅ `frame_features` + `processing_results` |
| `extract_features_from_npz.py`（本节） | 一帧一文件经过 `merge_xtherm_folder` → `detect_duplicate_frames` 后的 NPZ | ❌ 不写库，只产 CSV |

两条路径产出的 CSV 列名结构一致（`Tmax / Tmean / Tstd / Gmax / Gmean / Gstd / high_temp_area`），唯一差异是 NPZ 路径用 `frame` 列名而非 `frame_index`——下游论文图表脚本如果同时消费两种来源，只需统一一次列名即可。

---

## FAQ

**Q: probe 给出多个 offset 都"看起来合理"，怎么办？**
A: 选 `n_frames` 最大且温度范围最物理的；如果温度量纲全错（例如 Tmax 显示几万），多半是 `temperature_scale` 错了或 dtype 不是 `uint16`。

**Q: 我能改 `database/thermal_cladding.db` 的 schema 吗？**
A: 第一版尽量不改。如果一定要加列，用 `ALTER TABLE ... ADD COLUMN ...`；不要重命名 / 删除已有列，pipeline 与 6 个 CLI 都依赖现有列名。

**Q: 同一 `.xtherm` 文件被分析多次会污染数据库吗？**
A: 不会。`frame_features` 用 `UNIQUE(xtherm_file_id, frame_index) + INSERT OR REPLACE`，`processing_results` 用 `UNIQUE(xtherm_file_id) + ON CONFLICT DO UPDATE`。可安全重跑。
