import os
import time
import ctypes
from contextlib import suppress
from typing import Optional, List


def _env_flag(name: str, default: str = "0") -> bool:
    """
    辅助函数：解析布尔类型的环境变量。
    支持 "1", "true", "yes", "y", "on" 作为 True。
    """
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: str) -> int:
    """
    辅助函数：解析整数类型的环境变量。
    如果解析失败，返回默认值。
    """
    try:
        return int(os.getenv(name, default))
    except Exception:
        return int(default)


class ChakraTraceManager:
    """
    嵌入式 Tracer 管理器，用于收集 vLLM 的性能数据。
    
    核心功能：
      1. 收集 PyTorch Execution Trace (ET): 包含 CPU 算子及其依赖关系（用于生成计算图）。
      2. 收集 Kineto Trace: 包含 CPU/GPU 时间线（Timeline）和 CUDA Kernel 详细信息。

    新行为模式 (基于文件标志位的窗口控制):
      - 如果标志文件 (VLLM_CHAKRA_FLAG) 存在:
          调用 start() (仅一次) + 每轮迭代调用 step()
      - 如果标志文件不存在:
          调用 stop() (仅一次) -> 停止并导出当前窗口的所有 Trace 文件

    这种设计允许用户通过外部脚本（如 `touch /tmp/flag` -> 运行推理 -> `rm /tmp/flag`）
    来精确控制录制范围，例如只录制一个 `llm.generate()` 请求的全过程。
    """

    def __init__(self):
        # --- 功能开关与路径配置 ---
        # 总开关：是否启用 Chakra Trace
        self.enabled = _env_flag("VLLM_CHAKRA_TRACE", "1")
        # 输出目录：Trace 文件保存位置
        self.out_dir = os.getenv(
            "VLLM_CHAKRA_TRACE_DIR",
            "/home/ubuntu/vllm_quickstart/vllm_chakra",
        )

        # --- 窗口控制 ---
        # 标志文件路径：用于跨进程控制 Profile 的启停 (默认: /tmp/vllm_chakra_trace_on)
        self.flag_path = os.getenv("VLLM_CHAKRA_FLAG", "/tmp/vllm_chakra_trace_on")

        # --- 高级选项 ---
        # 是否自动转换 Trace 格式 (预留功能)
        self.auto_convert = _env_flag("VLLM_CHAKRA_AUTO_CONVERT", "0")
        # 是否记录 Tensor 的 Shape 信息 (对分析显存和计算量很有用)
        self.record_shapes = _env_flag("VLLM_CHAKRA_RECORD_SHAPES", "1")
        # 是否记录 Python 调用栈 (会增加开销，但有助于定位代码位置)
        self.with_stack = _env_flag("VLLM_CHAKRA_WITH_STACK", "0")
        # 是否在 Step 时强制同步 CUDA (会严重影响性能，仅用于调试)
        self.sync_cuda = _env_flag("VLLM_CHAKRA_SYNC_CUDA", "0")
        # 是否打印调试日志
        self.debug = _env_flag("VLLM_CHAKRA_DEBUG", "0")

        # --- 安全机制 ---
        # 自动停止阈值：即使标志文件未删除，跑了 N 步后也强制停止 (防止爆磁盘)
        self.max_steps = _env_int("VLLM_CHAKRA_TRACE_MAX_STEPS", "0")

        # --- 运行时状态 ---
        self._running = False  # 标记当前是否正在录制窗口中
        self._step = 0         # 当前窗口已录制的步数
        self._et = None        # ExecutionTraceObserver 实例
        self._prof = None      # torch.profiler.profile 实例

        # --- 输出文件路径 (在 start() 时动态生成) ---
        self.et_path: Optional[str] = None
        self.kineto_path: Optional[str] = None
        self.et_plus_path: Optional[str] = None
        self.chakra_out_dir: Optional[str] = None

        # --- 诊断信息 ---
        self._cuda_available: Optional[bool] = None
        self._cupti_loadable: Optional[bool] = None

    def _log(self, msg: str) -> None:
        """内部日志打印函数"""
        if self.debug:
            print(f"[ChakraTraceManager] {msg}", flush=True)

    def _pid_tag(self) -> str:
        """生成唯一的文件名后缀：pid + 时间戳，防止多进程覆盖"""
        return f"pid{os.getpid()}_{int(time.time()*1000)}"

    def _detect_cuda_and_cupti(self) -> None:
        """
        检测 CUDA 和 CUPTI (NVIDIA Profiling Interface) 是否可用。
        如果 CUPTI 库加载失败，将自动降级为仅 CPU Profile，防止程序崩溃。
        """
        import torch

        self._cuda_available = bool(torch.cuda.is_available())
        if not self._cuda_available:
            self._cupti_loadable = False
            self._log("CUDA not available; will run CPU-only profiling.")
            return

        try:
            # 尝试加载 CUPTI 动态库
            ctypes.CDLL("libcupti.so")
            self._cupti_loadable = True
            self._log("CUPTI load OK (libcupti.so). CUDA activity enabled.")
        except OSError as e:
            self._cupti_loadable = False
            self._log(
                "CUPTI load FAILED; CUDA activity will be disabled. "
                f"Reason: {e}. "
                "Hint: export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
            )

    def start(self) -> None:
        """
        开启一个新的 Profile 窗口。
        每个窗口对应一个独立的输出文件集 (ET json + Kineto json)。
        """
        if not self.enabled or self._running:
            return

        os.makedirs(self.out_dir, exist_ok=True)
        tag = self._pid_tag()

        # 定义输出路径
        self.et_path = os.path.join(self.out_dir, f"pytorch_et_{tag}.json")
        self.kineto_path = os.path.join(self.out_dir, f"kineto_{tag}.json")
        self.et_plus_path = os.path.join(self.out_dir, f"pytorch_et_plus_{tag}.json")
        self.chakra_out_dir = os.path.join(self.out_dir, f"chakra_trace_{tag}")

        self._detect_cuda_and_cupti()

        # 1. 启动 Execution Trace (ET)
        # 用于捕获算子依赖图
        self._et = None
        with suppress(Exception):
            from torch.profiler import ExecutionTraceObserver

            et = ExecutionTraceObserver()
            et.register_callback(self.et_path)
            et.start()
            self._et = et
            self._log(f"ExecutionTraceObserver started: {self.et_path}")

        # 2. 启动 Kineto Profiler (Timeline)
        # 注意：这里不使用 schedule，而是手动控制 step 和 stop
        from torch.profiler import profile, ProfilerActivity

        activities: List[ProfilerActivity] = [ProfilerActivity.CPU]
        # 只有在 CUDA 和 CUPTI 都可用时才启用 CUDA Profile
        if self._cuda_available and self._cupti_loadable:
            activities.append(ProfilerActivity.CUDA)

        self._prof = profile(
            activities=activities,
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
            # profile_memory=True,  # 需要分析显存泄漏时可打开
        )

        self._prof.__enter__()  # 手动进入上下文
        self._running = True
        self._step = 0
        self._log(f"Profiler started. activities={activities}")

    # 向后兼容的别名方法
    def maybe_start(self) -> None:
        self.start()

    def step(self) -> None:
        """
        记录一次迭代 (Iteration)。
        通常在 vLLM 的一次 execute_model 或 sample_tokens 结束时调用。
        此方法不会导出文件，只是推进一步 Profiler 的状态。
        """
        if not self._running:
            return

        if self.sync_cuda:
            with suppress(Exception):
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        try:
            self._prof.step()  # 标记当前 Step 结束
        except Exception as e:
            self._log(f"prof.step() failed: {e}")

        self._step += 1
        # 安全检查：如果超过最大步数，强制停止
        if self.max_steps > 0 and self._step >= self.max_steps:
            self._log(f"Reached max_steps={self.max_steps}, forcing stop().")
            self.stop()

    # 向后兼容的别名方法
    def step_and_maybe_stop(self) -> None:
        self.step()

    def stop(self) -> None:
        """
        停止当前 Profile 窗口并导出数据。
        这是生成 Trace 文件的唯一时刻。
        """
        if not self._running:
            return

        # 标记为已停止，防止重入
        self._running = False

        # 1. 先停止 ET (Execution Trace)
        if self._et is not None:
            try:
                self._et.stop()
            except Exception as e:
                self._log(f"ExecutionTraceObserver.stop() failed: {e}")
            try:
                self._et.unregister_callback()
            except Exception as e:
                self._log(f"ExecutionTraceObserver.unregister_callback() failed: {e}")
            self._et = None
            self._log("ExecutionTraceObserver stopped.")

        # 2. 再停止 Kineto Profiler 并导出数据
        # 重要：必须先调用 __exit__ (触发内部 stop)，然后才能 export
        if self._prof is not None:
            try:
                if self.sync_cuda:
                    with suppress(Exception):
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                # A) 停止 Profiler
                self._prof.__exit__(None, None, None)

                # B) 导出 Chrome Trace (JSON)
                if self.kineto_path:
                    self._log(f"Exporting chrome trace to: {self.kineto_path}")
                    self._prof.export_chrome_trace(self.kineto_path)
                    self._log(f"Kineto(chrome) trace exported: {self.kineto_path}")
                else:
                    self._log("kineto_path is None; skip export.")
            except BaseException as e:
                # 使用 BaseException 捕获所有异常（包括 SystemExit），确保在程序退出时也能打印错误
                import traceback
                self._log(f"Profiler stop/export FAILED: {e}")
                traceback.print_exc()
            finally:
                self._prof = None

        self._log("Profiler stopped.")

    def tick_by_flag(self) -> None:
        """
        核心驱动方法：在 Worker 的每次迭代边界调用。
        
        逻辑：
        - 检查标志文件是否存在。
        - 存在 -> 确保已 Start，并执行 Step。
        - 不存在 -> 确保已 Stop。
        """
        if not self.enabled:
            return

        if os.path.exists(self.flag_path):
            self.start()  # 如果已经在运行，start() 内部会直接返回
            self.step()
        else:
            self.stop()   # 如果已经停止，stop() 内部会直接返回


_global_mgr: Optional[ChakraTraceManager] = None


def get_global_manager() -> ChakraTraceManager:
    """
    获取进程级全局单例。
    确保每个进程（每个 GPU Worker）只有一个 Manager 实例。
    """
    global _global_mgr
    if _global_mgr is None:
        _global_mgr = ChakraTraceManager()
    return _global_mgr
