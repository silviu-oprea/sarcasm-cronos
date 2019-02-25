import logging


def create_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logging_formatter = logging.Formatter(
        '[%(asctime)s - %(name)s - %(levelname)s] %(message)s',
        '%Y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging_formatter)
    logger.addHandler(ch)

    return logger
