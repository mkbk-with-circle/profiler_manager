
# Profiler Manager (Chakra Trace Manager)

这是一个轻量的第三方 Python 库，用来在 **训练/推理** 过程中进行 profiling，采集：

- **PyTorch Execution Trace (ET)**：CPU op 与依赖关系
- **Kineto Trace**：CPU/GPU 时间线与 CUDA kernel（若 CUPTI 可用）

并通过 **“文件 Flag 控制的采样窗口”**，让你可以精确覆盖某一个推理/训练区间（例如“只 profile 一次 `llm.generate()`”），最终产出便于后续使用 Chakra 做 link、生成通用 workload 的 trace 文件。

---

## ✨ 核心特性

- **跨进程窗口控制**：通过一个 flag 文件控制 `start/step/stop`，适合多 worker/多进程场景
- **一次窗口 = 一次导出**：flag 存在时持续 `step()`；flag 消失时只导出一次（每进程一个文件）
- **GPU 可选**：自动检测 CUDA 与 `libcupti.so`，CUPTI 不可用时自动降级为 CPU-only
- **低侵入接入**：worker 主循环每步调用一次 `tick_by_flag()` 即可
- **环境变量可控**：输出目录、是否记录 shape、stack、强制 CUDA synchronize、最大步数等

---

## 📦 安装

如果这是一个带 `pyproject.toml` 的包，建议用 editable 安装（开发/改代码最方便）：

```bash
pip install -e .
```

或普通安装：

```bash
pip install .
```

---

## 🚀 快速开始

### 1) 在 worker/训练循环里“每步 tick 一次”

```python
from chakra_trace.manager import get_global_manager

mgr = get_global_manager()

for step in range(num_steps):
    # 你的训练/推理逻辑...
    mgr.tick_by_flag()
```

> `get_global_manager()` 是进程级单例：**每个进程一个 profiler 实例**。

---

### 2) 用 flag 文件圈定一个 profiling 窗口（只覆盖一次 generate）

你的控制端（driver）在调用目标区间前后创建/删除 flag 文件：

```bash
# 开始采样窗口
touch /tmp/vllm_chakra_trace_on

# 执行你想 profile 的那段（例如一次推理）
python run_infer.py

# 结束采样窗口（触发各进程 stop + 导出）
rm -f /tmp/vllm_chakra_trace_on
```

在 vLLM 场景，你也可以把 `touch/rm` 包在一次 `llm.generate()` 前后（driver 侧），worker 侧持续 `tick_by_flag()`。

---

## 🧠 工作机制（非常重要）

库的行为由 `tick_by_flag()` 驱动：

- **如果 flag 文件存在**：
  - `start()`（仅第一次）
  - 每次迭代执行 `step()`（只记录，不导出）
- **如果 flag 文件不存在**：
  - `stop()`（仅第一次）
  - 导出 **一个窗口对应的 trace 文件（每进程一份）**

这意味着你可以通过“touch/rm”精准控制 trace 覆盖范围，避免 schedule 方式导致导出多个碎片文件。

---

## ⚙️ 环境变量配置

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `VLLM_CHAKRA_TRACE` | `1` | 是否启用（`0/false` 关闭） |
| `VLLM_CHAKRA_TRACE_DIR` | `/home/ubuntu/vllm_quickstart/vllm_chakra` | 输出目录 |
| `VLLM_CHAKRA_FLAG` | `/tmp/vllm_chakra_trace_on` | 用于控制窗口的 flag 文件路径 |
| `VLLM_CHAKRA_AUTO_CONVERT` | `0` | 预留：自动转换（当前代码未使用） |
| `VLLM_CHAKRA_RECORD_SHAPES` | `1` | Kineto 是否记录 shape |
| `VLLM_CHAKRA_WITH_STACK` | `0` | 是否记录调用栈（开销更大） |
| `VLLM_CHAKRA_SYNC_CUDA` | `0` | 每步是否 `torch.cuda.synchronize()`（更准但更慢） |
| `VLLM_CHAKRA_DEBUG` | `0` | 打印调试日志 |
| `VLLM_CHAKRA_TRACE_MAX_STEPS` | `0` | 安全阀：最多 step 数；>0 时达到后强制 stop |

---

## 📁 输出文件说明

每个 profiling 窗口开始时会生成一组带 PID+时间戳的路径（每进程不同）：

- `pytorch_et_<tag>.json`：Execution Trace（ET）
- `kineto_<tag>.json`：Chrome Trace（Kineto 导出）
- `pytorch_et_plus_<tag>.json`：预留（当前未写入）
- `chakra_trace_<tag>/`：预留输出目录（当前未写入）

`<tag>` 形如：`pid12345_1700000000000`

---

## 🧰 GPU tracing 依赖（CUPTI）

要采集 CUDA kernel 活动，需要 `libcupti.so` 可被加载。若日志提示 CUPTI load failed，可尝试：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

当 CUDA 不可用或 CUPTI 不可用时，会自动变为 **CPU-only profiling**。

---

## ⚠️ 性能与文件体积建议

- `VLLM_CHAKRA_WITH_STACK=1`、`profile_memory`（如未来开启）会显著增加开销和 trace 体积
- 如果你只想覆盖一个很短的区间，使用 flag 窗口控制能有效减少文件大小
- 多进程/多 worker 下会导出多份文件（每进程一份），这是预期行为

---

## 🧩 API 一览

- `get_global_manager()`：获取进程级单例
- `ChakraTraceManager.start()`：开始一个窗口（手动模式）
- `ChakraTraceManager.step()`：记录一次迭代（不导出）
- `ChakraTraceManager.stop()`：停止并导出一次
- `ChakraTraceManager.tick_by_flag()`：推荐使用的自动模式（按 flag 控制窗口）

---

## 许可与声明

这是一个第三方工具库，用于 profiling 与 trace 生成；请在合规前提下使用，并注意不要将包含敏感信息的 trace 文件上传到公共仓库。
