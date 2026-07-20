# Triton FlashAttention-2 Forward 实验

本仓库提供一套用于分析 Triton FlashAttention-2 风格前向算子的可复现实验代码。实现将 `QKᵀ`、Online Softmax 和 `PV` 融合在一个 Triton kernel 中，不把完整注意力矩阵写回 HBM；配套代码负责正确性检查、四类参数扫描、理论 FLOPs/流量模型、NCU 命令生成与离线绘图。

仓库不包含预先生成的 benchmark 或 NCU 结果，也不会自动运行实验。

## 支持范围

- CUDA GPU 上的 dense self-attention forward
- 非 causal，`N_Q == N_K`
- 输入布局 `[B, H, N, D]`
- FP16/BF16 输入和同类型输出
- FP32 Online Softmax 状态与 FP32 dot/output 累加器
- Head Dimension `32、64、128`，实验性支持 `256`
- 不支持 backward、dropout、variable length、GQA、paged attention

计算定义为：

```python
scores = Q @ K.transpose(-1, -2) * sm_scale
probs = softmax(scores, dim=-1)
output = probs @ V
```

默认 `sm_scale = 1 / sqrt(D)`。

## 文件说明

- `triton_flash_attention.py`：Triton fused forward kernel 和配置对象。
- `attention_providers.py`：`torch_explicit`、`torch_sdpa`、`triton_fa2` 统一接口。
- `attention_models.py`：FLOPs、program 数、理论流量和算术强度模型。
- `experiment_configs.py`：默认扫描网格和调优候选配置。
- `test_flash_attention_correctness.py`：FP32 PyTorch reference 正确性测试。
- `benchmark_flash_attention.py`：单点、四类扫描、计时、显存统计和 CSV 输出。
- `generate_ncu_commands.py`：只生成 NCU 命令，不执行 NCU。
- `parse_ncu_csv.py`：归一化 NCU raw CSV，并可与 benchmark CSV 合并。
- `plot_flash_attention_results.py`：只读取 CSV 并生成图片。

## 依赖

需要兼容 CUDA 的 Python 环境以及：

- PyTorch（需提供 CUDA 和 `scaled_dot_product_attention`）
- Triton
- pandas
- matplotlib
- 可选：Nsight Compute CLI (`ncu`)，仅在用户手动 profile 时需要

正确性、benchmark、NCU 和绘图脚本不会自动安装依赖；只有用户明确提交的环境配置作业会执行安装。Triton kernel API、PyTorch SDPA 后端和 NCU 指标名称可能随版本变化；CSV 会记录 PyTorch/Triton 版本。

### 使用 Slurm 配置隔离环境

环境配置脚本会申请一张 H100，在仓库内创建 `.venv-fa2`，安装依赖并检查 PyTorch/Triton 导入与 CUDA 可见性。它不会运行正确性测试、benchmark 或 NCU：

两个 Slurm 脚本都会在 `/home/spack/spack/share/spack/setup-env.sh` 存在时加载 `cmake@3.28.6`、`cuda@12.9.0` 和 `python@3.13.0`。如果该文件不存在，脚本不会调用 Spack，并继续使用提交环境中的命令。

```bash
sbatch setup_flash_attention_env.sbatch
```

如果集群需要指定基础 Python：

```bash
FA2_BASE_PYTHON=/path/to/python3 sbatch setup_flash_attention_env.sbatch
```

如果需要指定 PyTorch CUDA wheel 源，例如 CUDA 12.8 wheel：

```bash
FA2_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
sbatch setup_flash_attention_env.sbatch
```

默认不固定 CUDA wheel 源，由 pip 当前配置决定。应根据集群 NVIDIA Driver、CUDA 兼容性和内部镜像调整 `FA2_TORCH_INDEX_URL`。配置日志写入 `fa2env.out`，精确包版本保存到 `.venv-fa2/requirements.lock.txt`。

配置成功后提交实验：

```bash
FA2_PYTHON="$PWD/.venv-fa2/bin/python" \
sbatch run_flash_attention_experiment.sbatch
```

## 正确性测试

先运行较小集合：

```bash
python test_flash_attention_correctness.py --quick
```

再运行完整覆盖：

```bash
python test_flash_attention_correctness.py
```

完整集合覆盖 `B={1,2}`、`H={1,4,8}`、`N={64,127,128,257,512}`、`D={32,64,128}` 和 FP16/BF16。`127`、`257` 专门覆盖非 Tile 整数倍。Reference 将输入转为 FP32 后计算完整 attention；每个 case 检查 `torch.allclose`、最大绝对/相对误差和 NaN/Inf。可以用 `--batch-sizes`、`--num-heads`、`--seq-lens`、`--head-dims`、`--dtypes` 缩小范围。

## Provider

统一接口为：

```python
output = run_attention(provider, q, k, v, sm_scale, config=None)
```

- `torch_explicit` 显式创建 `[B,H,N,N]` score 和 probability 中间矩阵。
- `torch_sdpa` 调用 PyTorch SDPA，实际使用 math、memory-efficient 或 flash 等哪一种后端由 PyTorch、设备、dtype 和 shape 决定，本项目不假定它一定是 FlashAttention。
- `triton_fa2` 每个 program 计算一个 Query tile，流式遍历 K/V tile。

## Benchmark

所有命令都会实际运行 GPU 实验，应由用户在目标机器上手动执行。默认使用固定配置 `BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=3`；通过 `--config-mode tuned` 对每个 shape 独立扫描经过过滤的候选配置，并记录最终胜出项。该调优是 Python 层逐项计时，行为易于复现，但耗时明显更多。

### Slurm 一键运行

仓库提供 H100 单卡脚本，依次执行完整正确性测试、四类 benchmark 和离线绘图，不运行 NCU：

```bash
sbatch run_flash_attention_experiment.sbatch
```

脚本默认使用 `python`。如果环境未通过提交 shell 继承，可指定解释器：

```bash
FA2_PYTHON=/path/to/environment/bin/python sbatch run_flash_attention_experiment.sbatch
```

结果写入 `results/slurm_$SLURM_JOB_ID/`。正确性失败会停止任务；benchmark 的局部 OOM、编译失败或运行失败会保存在 CSV 中并继续后续扫描。Slurm 标准输出写入 `fa2fwd.out`。

### A. 序列长度扫描

```bash
python benchmark_flash_attention.py \
  --experiment sequence \
  --providers torch_explicit torch_sdpa triton_fa2 \
  --measure-memory \
  --output-dir results
```

默认 `B=1, H=32, N={256,512,1024,2048,4096,8192}, D={64,128}`，覆盖 FP16/BF16。显式 Attention 在长序列很可能 OOM；该点会写成 `status=oom`，清理 cache 后继续。

### B. Head Dimension 扫描

固定配置：

```bash
python benchmark_flash_attention.py \
  --experiment head_dim \
  --provider triton_fa2 \
  --config-mode fixed \
  --block-m 64 --block-n 64 --num-warps 4 --num-stages 3
```

逐 shape 调优：

```bash
python benchmark_flash_attention.py \
  --experiment head_dim \
  --provider triton_fa2 \
  --config-mode tuned
```

默认 `B=1, H=32, N=4096, D={32,64,128}`。

### C. 并行度扫描

```bash
python benchmark_flash_attention.py \
  --experiment parallelism \
  --providers torch_sdpa triton_fa2
```

默认 `B=1, N=512, D=64, H={1,2,4,8,16,32,64}`，记录 `num_programs` 和 `programs_per_sm`。SM 数默认从当前设备查询，也可通过 `--sm-count` 明确传入。

### D. Tile 配置扫描

```bash
python benchmark_flash_attention.py \
  --experiment tile \
  --dtypes float16 bfloat16
```

默认扫描三个代表 shape 以及 `BLOCK_M={64,128}`、`BLOCK_N={32,64,128}`、`num_warps={4,8}`、`num_stages={2,3,4}`。单项编译/资源失败记为 `compile_error`，其他运行失败记为 `runtime_error`，不会终止整个扫描。

### 单 Shape 与计时选项

```bash
python benchmark_flash_attention.py \
  --experiment single \
  --provider triton_fa2 \
  --batch-size 1 --num-heads 32 --seq-len 4096 --head-dim 64 \
  --dtype float16 \
  --config-mode fixed \
  --block-m 64 --block-n 64 --num-warps 4 --num-stages 3 \
  --warmup 10 --repeat 50 --quantile 0.5
```

第一次 lazy/JIT 调用不计时，之后预热，再用 CUDA Events 对完整 Provider 调用计时并逐次同步。默认 `latency_ms` 是 median，同时固定记录 p20/p80；改变 `--quantile` 会改变 `latency_ms` 的主分位数，p20/p80 不变。`torch_explicit` 的延迟包括两个 matmul、缩放、softmax 以及所有相关中间操作，不是单一 matmul 时间。

`--measure-memory` 在计时完成后单独运行一次 Provider，并记录 reset 后的 peak。`peak_memory_reserved` 是 allocator 保留量，不等于真实 Tensor 占用；优先比较 `peak_memory_allocated`。Triton 寄存器和片上存储不会完整出现在 PyTorch allocated memory 中。

## 结果文件和字段

每次运行会在输出目录创建带实验名和微秒时间戳的 CSV 与 JSON；使用排他创建，不覆盖旧结果。JSON 保存原始命令和所有解析后的参数。

CSV 字段分组如下：

- 身份/状态：`experiment, provider, status, error_message, device_name, sm_count, torch_version, triton_version`
- shape：`dtype, batch_size, num_heads, seq_len, head_dim, causal`
- 配置：`block_m, block_n, num_warps, num_stages, config_mode`
- 计时：`warmup, repetitions, latency_ms, latency_p20_ms, latency_p80_ms, effective_tflops`
- 理论工作量：`qk_flops, pv_flops, total_matmul_flops, softmax_elements, matmul_flops_per_score`
- 并行度：`num_q_tiles, num_programs, programs_per_sm`
- 模型：`explicit_modeled_bytes, explicit_intermediate_bytes, fa_no_cache_modeled_bytes, modeled_arithmetic_intensity`
- 显存：`memory_allocated_before_bytes, peak_memory_allocated_bytes, peak_memory_reserved_bytes`

失败项保留 shape、配置、模型和完整异常类型/消息；不会填造延迟、TFLOPS 或显存数字。

## 理论模型

Non-causal forward 的矩阵乘 FLOPs：

```text
QK FLOPs    = 2 B H N² D
PV FLOPs    = 2 B H N² D
Total FLOPs = 4 B H N² D
```

Softmax 不强行折算为 Tensor Core FMA FLOPs，单独记录 `B H N²` 个元素；每个 score 对应 `4D` 个 QK+PV matmul FLOPs。

每个 Query tile 一个 program：

```text
num_q_tiles    = ceil(N / BLOCK_M)
num_programs   = B H num_q_tiles
programs_per_sm = num_programs / sm_count
```

显式 Attention 模型假设 Q/K/V 各读一次、O 写一次，并分别计算 score 写出/Softmax 读入、probability 写出/PV 读入。因此：

```text
explicit_intermediate_bytes = 4 B H N² intermediate_element_size
explicit_min_bytes = Q read + K read + V read + O write
                   + explicit_intermediate_bytes
```

这是简化的最低流量模型，未覆盖框架可能产生的额外临时量。

FlashAttention 无 cache 复用近似：

```text
kv_bytes_no_cache = B H ceil(N/BLOCK_M) 2 N D element_size
fa_no_cache_bytes = Q read + O write + kv_bytes_no_cache
```

该模型明确忽略不同 Query tile 之间的 L2 Cache 复用，是 K/V 重复读取的上界近似，不代表实际 DRAM Bytes。两种算术强度均以 `total_matmul_flops / modeled_bytes` 计算，后续应和 NCU 实测 DRAM/L2 指标对照。

可单独打印模型：

```bash
python attention_models.py --batch-size 1 --num-heads 32 --seq-len 4096 \
  --head-dim 64 --block-m 64 --dtype float16 --sm-count 120
```

这里的 SM 数仅作命令示例参数，应替换成目标设备查询值，不在代码中硬编码。

## Nsight Compute

脚本只生成命令，不会调用 NCU：

```bash
python generate_ncu_commands.py \
  --python python \
  --benchmark-script benchmark_flash_attention.py \
  --output-dir ncu_reports \
  --emit-csv \
  --commands-file run_ncu.sh
```

检查 `run_ncu.sh` 后由用户手动执行：

```bash
bash run_ncu.sh
```

默认 profile `N={512,2048,8192}` 与 `D={64,128}` 的六个点，固定 `B=1,H=32,provider=triton_fa2`。每个点生成独立 `.ncu-rep`；`--emit-csv` 会额外安排 raw CSV 采集。命令固定 tile/launch 配置，避免 profile 时 autotune。`--kernel-regex` 可适配不同 Triton 版本的 kernel 名称并排除输入初始化 kernel。

也可以先保存 `.ncu-rep`，再按本机 NCU 版本支持的方式导出 `--page raw --csv`。解析并与 benchmark CSV 合并：

```bash
python parse_ncu_csv.py ncu_reports/*.csv \
  --benchmark-csv results/single_TIMESTAMP.csv \
  --output results/ncu_merged.csv
```

NCU 不同版本的指标名称不完全一致。所有别名集中在 `parse_ncu_csv.py` 的 `METRIC_NAME_MAP`；可通过 `--metric-map custom_metrics.json` 增补。解析器预留 DRAM/L2 流量、L2 hit rate、寄存器、shared memory、occupancy、waves、warp scheduler、tensor pipe 和 local memory 字段。找不到的指标留空，不会伪造。

## 绘图

绘图脚本只读已有 CSV，绝不重新运行 benchmark：

```bash
python plot_flash_attention_results.py \
  results/sequence_TIMESTAMP.csv \
  results/parallelism_TIMESTAMP.csv \
  results/ncu_merged.csv \
  --output-dir plots
```

按数据可用性生成 sequence length–TFLOPS、sequence length–latency、peak memory、head dimension–寄存器/occupancy/TFLOPS、program 并行度、理论/实测流量及 L2 hit rate 图。缺少 NCU 列或没有有效行时跳过对应图片而不报错。

## Kernel 数据流与分析边界

每个二维 launch 中的 program 对应一个 `(query_tile, batch_head)`。Program 将 Q tile 加载到片上，循环读取 K/V tile：计算 `QKᵀ`，用当前 tile 行最大值更新 FP32 running max，按新最大值重缩放历史 FP32 normalization sum 和输出累加器，再将当前 probability tile 与 V 相乘并累加。循环结束后除以 running sum，转换为输出 dtype 并带尾部 mask 写回。完整 score/probability 矩阵从不写入 HBM。

融合 kernel 内部无法仅凭总执行时间精确拆分 QK、Softmax 和 PV 的耗时。本实验通过 Head Dimension 扫描、固定 tile 对照，以及 NCU 的 tensor pipe、occupancy、寄存器和 warp 调度指标间接分析非矩阵运算开销与资源约束。

## 已知限制与版本相关点

- 本仓库代码尚需在目标 CUDA GPU、具体 PyTorch/Triton 组合上实际编译和正确性验证。
- Triton 对 `tl.dot`、流水 stage、资源限制和生成 kernel 名称的行为可能随版本变化。
- `torch_sdpa` 的实际后端由 PyTorch 和设备决定；不应把该 Provider 自动标记为 FlashAttention。
- NCU metric 名称、单位和 CLI 选项会随 CUDA/NCU 版本变化，必要时扩展映射或调整生成命令。
- FP16/BF16 的误差容差是起始值，应依据目标硬件和 Triton 版本的实际正确性结果审阅，不能用放宽容差掩盖错误。
- 显式 Attention 在长序列下可能 OOM；这正是需要记录的实验现象之一。
- 当前没有 backward、causal、dropout、variable-length、GQA 或 paged attention。
