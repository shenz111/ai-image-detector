import os
import logging
from datetime import datetime


def setup_logger(save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, f"train_{datetime.now():%Y%m%d_%H%M%S}.log")

    logger = logging.getLogger("AIDetector")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger