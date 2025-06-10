"""Common logging setup."""
import logging
from pathlib import Path
from datetime import datetime

def get_logger(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    logger.setLevel(level)

    # Console output -------------------------------------------------------- #
    fmt = "[%(asctime)s] [%(levelname)s] %(name)s » %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Optional file output -------------------------------------------------- #
    logfile = Path("logs") / f"{datetime.now():%Y-%m-%d}.log"
    logfile.parent.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("Logger initialised")
    return logger
