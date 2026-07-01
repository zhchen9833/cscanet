# Copyright (c) OpenMMLab. All rights reserved.
import logging
import platform
import sys
import time
from os import mkdir
from os.path import dirname, exists

import torch.distributed as dist

logger_initialized: dict = {}


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:

            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')


def get_root_logger(log_file=None, log_level=logging.INFO):
    if log_file:
        log_dir = dirname(log_file)
        if not exists(log_dir):
            mkdir(log_dir)
    logger = get_logger(name='Uchiha', log_file=log_file, log_level=log_level)
    # info = get_env_info()
    # log_env_info(logger, info)
    return logger


def get_env_info():
    import torch
    import torchvision

    os_info = {
        "Python Version": sys.version.split()[0],
        "OS Platform": platform.platform()
    }

    torch_info = {
        "PyTorch Version": torch.__version__,
        "Torchvision Version": torchvision.__version__,
        "CUDA Available": torch.cuda.is_available()
    }

    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info.update({
            "GPU Model": torch.cuda.get_device_name(0),
            "GPU Memory": f"{round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)} GB",
            "CUDA Version": torch.version.cuda
        })
    else:
        gpu_info["GPU"] = "No CUDA-capable GPU detected"

    return {
        "System Info": os_info,
        "PyTorch Info": torch_info,
        "GPU Info": gpu_info
    }


def log_env_info(logger, info):
    logger.info(r"""
                           _  __    ____    __  __
                          | |/ /   / __ \  / / / /
                          |   /   / / / / / / / /
                         /   |   / /_/ / / /_/ /
                        /_/|_|  /_____/  \____/

   ______                     __           __                    __      __
  / ____/  ____   ____   ____/ /          / /   __  __  _____   / /__   / /
 / / __   / __ \ / __ \ / __  /          / /   / / / / / ___/  / //_/  / /
/ /_/ /  / /_/ // /_/ // /_/ /          / /___/ /_/ / / /__   / ,<    /_/
\____/   \____/ \____/ \__,_/          /_____/\__,_/  \___/  /_/|_|  (_)

    """)

    logger.info("ENVIRONMENT INFO".center(40))
    logger.info("=" * 40)

    for category, data in info.items():
        logger.info("=" * 10 + f"{category}" + "=" * 10)
        for k, v in data.items():
            logger.info(f"  {k + ':':<18} {v}")

    logger.info("" + "=" * 40)
    logger.info("ENVIRONMENT INFO".center(40))


class ETACalculator:
    def __init__(self, total_steps):
        self.start_time = time.time()
        self.total_steps = total_steps
        self.steps_completed = 0

    def update(self, steps=1):
        """更新已完成步数，返回剩余时间（秒）"""
        self.steps_completed += steps
        elapsed = time.time() - self.start_time
        time_per_step = elapsed / self.steps_completed
        remaining_steps = self.total_steps - self.steps_completed
        return remaining_steps * time_per_step

    def format_eta(self, seconds):
        """将秒数转换为 DDdHHhMMmSSs 格式（两位数标准格式）"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)

        parts = []
        if days > 0:
            parts.append(f"{days:02d}d")  # 两位数天
        if hours > 0 or days > 0:
            parts.append(f"{hours:02d}h")  # 两位数小时
        if minutes > 0 or hours > 0 or days > 0:
            parts.append(f"{minutes:02d}m")  # 两位数分钟
        parts.append(f"{seconds:02d}s")  # 两位数秒

        return "".join(parts)


if __name__ == "__main__":
    get_root_logger()
