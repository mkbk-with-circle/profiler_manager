import os
import time
import ctypes
from contextlib import suppress
from typing import Optional, List


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return int(default)


class ChakraTraceManager:
    """
    Embedded tracer for collecting:
      - PyTorch Execution Trace (ET): CPU ops + deps
      - Kineto trace: CPU/GPU timeline + CUDA kernels

    New behavior (flag-driven window):
      - If VLLM_CHAKRA_FLAG file exists:
          start() (once) + step() every iteration
      - If flag file does NOT exist:
          stop() (once) -> export a SINGLE trace file per process for that window

    This matches: "cover exactly one llm.generate()" by wrapping generate with
    touch/rm of the flag file, while workers tick_by_flag() each iteration.
    """

    def __init__(self):
        # Feature flags / paths
        self.enabled = _env_flag("VLLM_CHAKRA_TRACE", "1")
        self.out_dir = os.getenv(
            "VLLM_CHAKRA_TRACE_DIR",
            "/home/ubuntu/vllm_quickstart/vllm_chakra",
        )

        # Flag-driven window control (cross-process)
        self.flag_path = os.getenv("VLLM_CHAKRA_FLAG", "/tmp/vllm_chakra_trace_on")

        # Options
        self.auto_convert = _env_flag("VLLM_CHAKRA_AUTO_CONVERT", "0")
        self.record_shapes = _env_flag("VLLM_CHAKRA_RECORD_SHAPES", "1")
        self.with_stack = _env_flag("VLLM_CHAKRA_WITH_STACK", "0")
        self.sync_cuda = _env_flag("VLLM_CHAKRA_SYNC_CUDA", "0")
        self.debug = _env_flag("VLLM_CHAKRA_DEBUG", "0")

        # Optional safety: auto-stop after N steps even if flag not cleared (0 disables)
        self.max_steps = _env_int("VLLM_CHAKRA_TRACE_MAX_STEPS", "0")

        # Runtime state
        self._running = False
        self._step = 0
        self._et = None
        self._prof = None

        # Output paths (set on start)
        self.et_path: Optional[str] = None
        self.kineto_path: Optional[str] = None
        self.et_plus_path: Optional[str] = None
        self.chakra_out_dir: Optional[str] = None

        # Diagnostics
        self._cuda_available: Optional[bool] = None
        self._cupti_loadable: Optional[bool] = None

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[ChakraTraceManager] {msg}", flush=True)

    def _pid_tag(self) -> str:
        return f"pid{os.getpid()}_{int(time.time()*1000)}"

    def _detect_cuda_and_cupti(self) -> None:
        import torch

        self._cuda_available = bool(torch.cuda.is_available())
        if not self._cuda_available:
            self._cupti_loadable = False
            self._log("CUDA not available; will run CPU-only profiling.")
            return

        try:
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
        """Start a new profiling window (one window => one exported file per process)."""
        if not self.enabled or self._running:
            return

        os.makedirs(self.out_dir, exist_ok=True)
        tag = self._pid_tag()

        self.et_path = os.path.join(self.out_dir, f"pytorch_et_{tag}.json")
        self.kineto_path = os.path.join(self.out_dir, f"kineto_{tag}.json")
        self.et_plus_path = os.path.join(self.out_dir, f"pytorch_et_plus_{tag}.json")
        self.chakra_out_dir = os.path.join(self.out_dir, f"chakra_trace_{tag}")

        self._detect_cuda_and_cupti()

        # ET (best-effort)
        self._et = None
        with suppress(Exception):
            from torch.profiler import ExecutionTraceObserver

            et = ExecutionTraceObserver()
            et.register_callback(self.et_path)
            et.start()
            self._et = et
            self._log(f"ExecutionTraceObserver started: {self.et_path}")

        # Kineto profiler (NO schedule; we export explicitly on stop())
        from torch.profiler import profile, ProfilerActivity

        activities: List[ProfilerActivity] = [ProfilerActivity.CPU]
        if self._cuda_available and self._cupti_loadable:
            activities.append(ProfilerActivity.CUDA)

        self._prof = profile(
            activities=activities,
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
            # profile_memory=True,  # 需要的话可打开，但会增大开销/文件
        )

        self._prof.__enter__()
        self._running = True
        self._step = 0
        self._log(f"Profiler started. activities={activities}")

    # Backward-compatible name
    def maybe_start(self) -> None:
        self.start()

    def step(self) -> None:
        """Record one iteration. Does NOT export any file."""
        if not self._running:
            return

        if self.sync_cuda:
            with suppress(Exception):
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        try:
            self._prof.step()
        except Exception as e:
            self._log(f"prof.step() failed: {e}")

        self._step += 1
        if self.max_steps > 0 and self._step >= self.max_steps:
            self._log(f"Reached max_steps={self.max_steps}, forcing stop().")
            self.stop()

    # Backward-compatible name, but no longer auto-stops by schedule.
    def step_and_maybe_stop(self) -> None:
        self.step()

    def stop(self) -> None:
        """Stop current window and export exactly once."""
        if not self._running:
            return

        # Mark stopped early to avoid re-entrancy
        self._running = False

        # Stop ET first
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

        # IMPORTANT:
        # kineto_results is produced on profiler.stop(), which is triggered by __exit__.
        # If you export BEFORE __exit__, kineto_results may still be None
        # -> AttributeError: 'NoneType' object has no attribute 'save'
        if self._prof is not None:
            try:
                if self.sync_cuda:
                    with suppress(Exception):
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                # 1) finalize/stop the profiler first
                self._prof.__exit__(None, None, None)

                # 2) then export chrome trace
                if self.kineto_path:
                    self._log(f"Exporting chrome trace to: {self.kineto_path}")
                    self._prof.export_chrome_trace(self.kineto_path)
                    self._log(f"Kineto(chrome) trace exported: {self.kineto_path}")
                else:
                    self._log("kineto_path is None; skip export.")
            except BaseException as e:
                # Use BaseException so SystemExit during engine shutdown doesn't hide the root cause
                import traceback
                self._log(f"Profiler stop/export FAILED: {e}")
                traceback.print_exc()
            finally:
                self._prof = None

        self._log("Profiler stopped.")

    def tick_by_flag(self) -> None:
        """
        Call this at each iteration boundary inside worker.

        If flag exists: start (once) + step
        Else: stop (once)
        """
        if not self.enabled:
            return

        if os.path.exists(self.flag_path):
            self.start()
            self.step()
        else:
            self.stop()


_global_mgr: Optional[ChakraTraceManager] = None


def get_global_manager() -> ChakraTraceManager:
    """Process-global singleton (one profiler per process)."""
    global _global_mgr
    if _global_mgr is None:
        _global_mgr = ChakraTraceManager()
    return _global_mgr
