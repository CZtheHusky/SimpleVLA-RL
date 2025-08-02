    
import logging
from pathlib import Path
import os

class Logger:
    def __init__(self, log_dir="logs", log_name=None):
        pid = os.getpid()
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(log_dir, f"{log_name}_{pid}.log")

        self.logger = logging.getLogger(f"Logger_{pid}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # 不重复输出到控制台

        if not self.logger.handlers:
            formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S")

            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def log(self, msg: str):
        self.logger.info(msg)