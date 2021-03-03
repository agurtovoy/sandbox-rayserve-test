import logging
import sys

def configureLogger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    stdout_logging_handler = logging.StreamHandler(sys.stdout)
    stdout_logging_handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", datefmt="%m-%d %H:%M:%S"))
    logger.addHandler(stdout_logging_handler)

    return logger
