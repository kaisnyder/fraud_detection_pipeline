"""Lightweight timing context-manager."""
import time
from contextlib import contextmanager
from .logging import get_logger

logger = get_logger(__name__)

@contextmanager
def timeit(msg: str = "Operation"):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"{msg} finished in {elapsed:0.2f}s")
