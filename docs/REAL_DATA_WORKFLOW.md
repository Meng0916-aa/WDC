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

## FAQ

**Q: probe 给出多个 offset 都"看起来合理"，怎么办？**
A: 选 `n_frames` 最大且温度范围最物理的；如果温度量纲全错（例如 Tmax 显示几万），多半是 `temperature_scale` 错了或 dtype 不是 `uint16`。

**Q: 我能改 `database/thermal_cladding.db` 的 schema 吗？**
A: 第一版尽量不改。如果一定要加列，用 `ALTER TABLE ... ADD COLUMN ...`；不要重命名 / 删除已有列，pipeline 与 6 个 CLI 都依赖现有列名。

**Q: 同一 `.xtherm` 文件被分析多次会污染数据库吗？**
A: 不会。`frame_features` 用 `UNIQUE(xtherm_file_id, frame_index) + INSERT OR REPLACE`，`processing_results` 用 `UNIQUE(xtherm_file_id) + ON CONFLICT DO UPDATE`。可安全重跑。
