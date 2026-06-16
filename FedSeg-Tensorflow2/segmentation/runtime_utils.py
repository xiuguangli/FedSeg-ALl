from __future__ import annotations

import os
import re
import sys
import tempfile
import threading


_STDERR_FILTER_INSTALLED = False
_TF_NOISE_PATTERNS = (
    re.compile(rb"WARNING: All log messages before absl::InitializeLog\(\) is called"),
    re.compile(rb"I\d+\s+\S+\s+\d+\s+gpu_device\.cc:\d+\] Created device "),
    re.compile(rb"I\d+\s+\S+\s+\d+\s+device_compiler\.h:\d+\] Compiled cluster using XLA!"),
)


def configure_tensorflow_env(gpu: str | None = None) -> None:
    if gpu is not None:
        gpu_value = str(gpu).strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if gpu_value == "" else gpu_value

    # Keep TensorFlow backend logs quiet unless the user explicitly overrides them.
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "fedseg-matplotlib"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


def install_tensorflow_stderr_filter() -> None:
    global _STDERR_FILTER_INSTALLED
    if _STDERR_FILTER_INSTALLED:
        return

    try:
        original_stderr_fd = os.dup(2)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 2)
        os.close(write_fd)
    except OSError:
        return

    def _forward_filtered_stderr() -> None:
        buffer = b""
        try:
            while True:
                chunk = os.read(read_fd, 4096)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    raw = line + b"\n"
                    if any(pattern.search(raw) for pattern in _TF_NOISE_PATTERNS):
                        continue
                    os.write(original_stderr_fd, raw)
            if buffer and not any(pattern.search(buffer) for pattern in _TF_NOISE_PATTERNS):
                os.write(original_stderr_fd, buffer)
        except OSError:
            pass
        finally:
            try:
                os.close(read_fd)
            except OSError:
                pass
            try:
                os.close(original_stderr_fd)
            except OSError:
                pass

    thread = threading.Thread(target=_forward_filtered_stderr, name="fedseg-tf-stderr-filter", daemon=True)
    thread.start()
    _STDERR_FILTER_INSTALLED = True


def configure_tensorflow_runtime(tf) -> None:
    try:
        tf.get_logger().setLevel("ERROR")
    except Exception:
        pass

    try:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except Exception:
        pass

    try:
        tf.autograph.set_verbosity(0)
    except Exception:
        pass

    try:
        tf.config.optimizer.set_jit(False)
        options = tf.config.optimizer.get_experimental_options()
        options["layout_optimizer"] = False
        tf.config.optimizer.set_experimental_options(options)
    except Exception:
        pass

    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
        absl.logging.set_stderrthreshold("error")
    except Exception:
        pass


def should_disable_tqdm() -> bool:
    override = os.environ.get("FEDSEG_DISABLE_TQDM")
    if override is not None:
        return override.strip().lower() not in {"0", "false", "no", "off"}
    return False


def should_use_xla() -> bool:
    override = os.environ.get("FEDSEG_USE_XLA")
    if override is not None:
        return override.strip().lower() not in {"0", "false", "no", "off"}
    return False
