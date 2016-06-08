import logging
import sys

def build_logger():
    reload(logging)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    return logger
