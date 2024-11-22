import logging
import os

#: The logger that will be used to log the execution information
execution_logger = logging.getLogger()


def setup_logging(
    log_file=None,
    datefmt="%m/%d/%Y %H:%M:%S %p",
    fmt="%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s",
    level=logging.INFO,
    name="logger",
):
    """Setup the logging configuration.

    :param log_file: The log file path, defaults to None
    :type log_file: str, optional
    :param datefmt: The date format, defaults to "%m/%d/%Y %H:%M:%S %p"
    :type datefmt: str, optional
    :param fmt: The log format, defaults to "%(asctime)s [%(name)s] [%(levelname)-5.5s]  %(message)s"
    :type fmt: str, optional
    :param level: The log level, defaults to logging.INFO
    :type level: int, optional
    :param name: The logger name, defaults to "logger"
    :type name: str, optional
    """
    execution_logger.name = name

    execution_logger.handlers.clear()
    execution_logger.setLevel(level)

    log_formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    execution_logger.addHandler(console_handler)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_formatter)
        execution_logger.addHandler(file_handler)
