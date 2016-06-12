import logging
import sys

def build_logger(head):
    reload(logging)
    logging.basicConfig(level=logging.DEBUG, format=head)
    logger = logging.getLogger()
    return logger
