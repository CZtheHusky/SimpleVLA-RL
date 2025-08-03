import logging
import os
from pathlib import Path

class ContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        # 合并初始化时的 extra 和每次调用时传入的 extra
        extra = self.extra.copy()
        extra.update(kwargs.get("extra", {}))
        kwargs["extra"] = extra
        return msg, kwargs

class DummyLogger:
    def __init__(self, *args, **kwargs): pass

    def log(self, *args, **kwargs): pass

    def warning(self, *args, **kwargs): pass

class LocalLogger:
    def __init__(self, log_dir="logs", log_name=None, world_size=None, rank=None):
        pid = os.getpid()
        log_dir = os.path.join(log_dir, log_name) if log_name else log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(log_dir, f"{pid}.log")

        # 1) 创建基础 logger
        base_logger = logging.getLogger(f"Logger_{pid}")
        base_logger.setLevel(logging.INFO)
        base_logger.propagate = False  # 不向 root logger 传递

        if not base_logger.handlers:
            # 2) 在 format 中加上 %(world_size)s 和 %(rank)s
            fmt = "%(asctime)s.%(msecs)03d [ws=%(world_size)s rank=%(rank)s] %(message)s"
            datefmt = "%Y-%m-%d %H:%M:%S"
            formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            base_logger.addHandler(fh)

        # 3) 用 Adapter 注入固定的 extra
        extra = {
            "world_size": world_size if world_size is not None else "",
            "rank":      rank      if rank      is not None else ""
        }
        self.logger = ContextAdapter(base_logger, extra)

    def log(self, msg: str):
        # 直接调用，就会自动带上 world_size 和 rank
        self.logger.info(msg)
