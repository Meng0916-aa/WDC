# GEJ-WDC — 稳恒磁场辅助多道激光熔覆温度场分析

CoCrNi 粉末在 316L 不锈钢基板上、在 0/40/80/120 mT 稳恒磁场下多道激光熔覆的红外温度场（Xiris VXIR-3000，`.xtherm`）数据库驱动处理框架。

> 当前没有真实实验数据；代码全部按真实实验数据的工作流设计。`tests/` 使用极小的临时二进制 fixture 保证可独立运行，但不充当生产数据流水线。

## 项目结构

```
D:\GEJ-WDC\
├── CLAUDE.md
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml            # VXIR-3000 默认参数 + 实验参数 + 处理参数
├── database/
│   └── thermal_cladding.db     # 由 init_database.py 创建, 仅存路径与元数据
├── data/
│   ├── raw/                    # 真实 .xtherm 放这里 (按 B-mT/sample 分组)
│   ├── processed/              # 处理中间产物
│   └── features/               # 单文件特征 CSV + 汇总 CSV
├── docs/
│   └── REAL_DATA_WORKFLOW.md   # 真实数据接入详细 walkthrough (含排错)
├── results/figures/            # 后续论文图表输出位置
├── src/
│   ├── utils/                  # 配置 / 路径 / 日志
│   ├── db/                     # SQLite schema + 登记 + 查询
│   ├── io/                     # xtherm 读取 + 格式探测
│   ├── processing/             # 温度梯度 + 特征
│   └── pipeline/               # 数据库驱动的处理流程
├── scripts/                    # 7 个 CLI (含 check_ready_for_real_data.py)
└── tests/                      # pytest 单元测试 (49 cases)
```

## 安装

```bash
pip install -r requirements.txt
```

依赖：numpy / scipy / pandas / pyyaml / tqdm / pytest / matplotlib。

---

## 真实数据接入工作流

### Step 0: 真实数据接入前检查

```bash
python scripts/check_ready_for_real_data.py --config configs/default.yaml
```

退出码：

- **`exit_code = 0`** —— 项目已经完全就绪，可以登记、探测、计算温度梯度和批量处理。
- **`exit_code = 1`** —— 项目可以登记 `.xtherm` 文件，也可以进行格式探测，但**不能计算温度梯度**。通常原因是 `dx_mm_per_pixel` 和 `dy_mm_per_pixel` 还没有填写。**此时不要运行 `batch_process_database.py`**。
- **`exit_code = 2`** —— 项目基础结构不完整，例如数据库、目录、配置或 schema 缺失。需要先运行 `init_database.py` 或修复项目结构。

### 推荐流程（真实数据到来后的执行顺序）

1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

2. **初始化数据库**：
   ```bash
   python scripts/init_database.py --config configs/default.yaml
   ```

3. **运行预检查**：
   ```bash
   python scripts/check_ready_for_real_data.py --config configs/default.yaml
   ```

4. **如果 `exit_code=1`**，说明可以登记和探测，但不能计算梯度。此时可以继续登记真实 `.xtherm` 文件，**但不要批量处理**。

5. **登记真实 `.xtherm` 文件**：
   ```bash
   python scripts/register_xtherm_files.py --config configs/default.yaml \
       --input data/raw/B000/sample01 --sample-id B000_sample01 --B-mT 0 --recursive
   ```

6. **探测第一个真实文件格式**：
   ```bash
   python scripts/probe_registered_file.py --config configs/default.yaml --file-id 1
   ```

7. **如果 `header_offset`、帧数、温度范围合理，则写回数据库**：
   ```bash
   python scripts/probe_registered_file.py --config configs/default.yaml --file-id 1 --apply
   ```

8. **完成空间标定后，填写 `configs/default.yaml` 中**：
   ```yaml
   camera:
     dx_mm_per_pixel: <靶标实测>
     dy_mm_per_pixel: <靶标实测>
   ```

9. **再次运行预检查**：
   ```bash
   python scripts/check_ready_for_real_data.py --config configs/default.yaml
   ```

10. **只有当 `exit_code=0` 时，才运行单文件处理**：
    ```bash
    python scripts/process_registered_file.py --config configs/default.yaml --file-id 1
    ```

11. **单文件处理结果合理后，再运行批量处理**：
    ```bash
    python scripts/batch_process_database.py --config configs/default.yaml --status registered
    # 或者
    python scripts/batch_process_database.py --config configs/default.yaml --status probed
    ```

12. **导出总特征表**：
    ```bash
    python scripts/export_features_from_database.py --config configs/default.yaml
    ```

13. **运行测试**：
    ```bash
    python -m pytest tests/
    ```

> 每一步的预期输出、典型故障与排查、相机模式（AOI / 1000 fps / 不同 fps）的配置改动以及 `status='error'` 恢复流程，详见 [`docs/REAL_DATA_WORKFLOW.md`](docs/REAL_DATA_WORKFLOW.md)。

---

## Important notes before real data processing

1. **原始 `.xtherm` 文件不要存入数据库**——数据库只保存路径（相对项目根）和元数据；原始二进制始终留在 `data/raw/` 文件系统里。
2. **真实数据到来后，必须先 probe `header_offset`，不要直接批处理**——`configs/default.yaml` 里的 `header_offset=0` 是默认占位，Xiris 不同导出模式可能有非零文件头。
3. **`dx_mm_per_pixel` 和 `dy_mm_per_pixel` 是计算 `degC/mm` 温度梯度的必要条件**——这两个值来自相机标定（已知尺寸靶标 / 工件像素跨度换算），不能猜，不能用默认。
4. **如果 dx/dy 没有标定，只能做文件登记和格式探测**；任何 `compute_gradients` 调用会立刻抛 `ValueError`，pipeline 会把该文件记 `error`。预检查脚本退出码 `1` 就是这种状态。
5. **如果相机使用 AOI 或 1000 fps 模式，必须修改 `width`、`height`、`fps`**——AOI 改了像素阵列形状，1000 fps 通常伴随像素阵列裁切，两者都会让 `frame_bytes` 不匹配。
6. **如果实际采集帧率不是 180 fps，需要修改 `fps`**，否则 `frame_features.time_s = frame_index / fps` 这条时间轴会错位，所有后续基于时间的分析都跟着错。
7. **当前 `temperature_scale` 默认为 `0.1`**，即 `raw_value × 0.1 = degC`；如果 Xiris 实际导出配置是另一刻度，必须在 YAML 中改这个值。
8. **当前阶段不处理 emissivity / transmissivity 的后处理重映射**——默认 Xiris / WeldStudio 已按工件设定的发射率输出温度数据；如果以后需要后处理发射率修正，会新增一个独立标定模块，不在本框架第一版范围。

---

## 关键设计约束

1. **没有 synthetic experiment pipeline**：生产路径只跑数据库里登记过的真实 `.xtherm`。任何 fixture 都仅限 `tests/` 目录使用，且容量极小。
2. **不存原始二进制**：`xtherm_files` 表只保留路径（相对项目根）与元数据。
3. **不允许 silent reshape**：`(file_size - header_offset)` 必须被 `frame_bytes` 整除；不整除时 reader 抛出包含文件大小、帧字节数、剩余字节数的明确错误信息。
4. **缺标定就拒绝跑梯度**：`dx_mm_per_pixel` 与 `dy_mm_per_pixel` 是相机标定结果；缺失时 `compute_gradients` 抛 `ValueError` 而不是猜值。
5. **状态机驱动**：`xtherm_files.status` 由 `registered → probed → processed | error` 推进，所有 CLI 都基于这一状态过滤。
6. **路径相对项目根**：数据库内不存绝对路径，跨机器迁移友好。
7. **当前阶段不写深度学习模型**：先把数据库 / 读取 / 探测 / 梯度 / 特征做扎实，PyTorch 时序模型留给下一阶段。

## 还缺哪些真实实验参数（在跑真实数据前必须补齐）

- **`dx_mm_per_pixel` / `dy_mm_per_pixel`**：相机标定得到的像素到毫米换算系数。`configs/default.yaml` 当前为 `null`；跑真实数据前必须用靶标标定测得，写入 YAML。
- **真实 `header_offset` 与 `(width, height)`**：登记后立刻跑 `probe_registered_file.py`，再 `--apply` 把 best 候选写回。
- **`temperature_scale` 与温度量纲**：默认 `0.1`；从 Xiris 实际导出配置确认。
- **`fps` 真实测量值**：默认 `180 Hz`；不一致时 `time_s` 会错。
- **`emissivity` 后处理**：当前框架假设 Xiris 软件已按工件设定发射率换算好温度；如需重映射需新增标定模块。

## 错误处理与状态机

| status | 触发 |
| --- | --- |
| `registered` | 由 `register_xtherm_files.py` 登记完成 |
| `probed` | 由 `probe_registered_file.py --apply` 应用了 best 候选 |
| `processed` | 由 `process_registered_file.py` / `batch_process_database.py` 成功完成 |
| `error` | 处理失败；`processing_results.error_message` 中保留异常类型 + 信息 + traceback 摘要 |

失败的文件会留在 `error` 状态，可在修复参数后再跑一次（pipeline 内部使用 `INSERT OR REPLACE`，可安全重跑）。具体恢复步骤见 [`docs/REAL_DATA_WORKFLOW.md`](docs/REAL_DATA_WORKFLOW.md)。
