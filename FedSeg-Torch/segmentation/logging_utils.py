from datetime import datetime
from pathlib import Path
import sys

from loguru import logger
from tqdm import tqdm


def _tqdm_sink(message):
    # 通过 tqdm.write 输出日志，避免进度条刷新时把终端内容打乱。
    tqdm.write(message.rstrip(), file=sys.stderr)


def setup_logger(verbose=False, logs_dir="logs", log_name="train"):
    level = "DEBUG" if verbose else "INFO"
    log_dir = Path(logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = log_dir / f"{log_name}_{timestamp}.log"
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "{message}"
    )

    # 每次显式重建 sink，保证终端输出和文件输出使用同一套格式，不与旧配置叠加。
    logger.remove()
    logger.add(
        _tqdm_sink,
        level=level,
        format=log_format,
        colorize=True,
        enqueue=False,
    )
    logger.add(
        str(log_path),
        level=level,
        format=log_format,
        colorize=False,
        enqueue=False,
    )
    return logger


__all__ = ["logger", "setup_logger"]
