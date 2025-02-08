import logging
from logging.handlers import TimedRotatingFileHandler

LOGGER_NAME:str="global_logger"
LOG_FILE:str=None
formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s %(levelname)s %(message)s')
logger = logging.getLogger(LOGGER_NAME)  # Use a consistent logger name
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set console logging level
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
    
def config_logger(logger_name:str,output2console=True):
    LOGGER_NAME = logger_name
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s %(levelname)s %(message)s')
    logger = logging.getLogger(LOGGER_NAME)  # Use a consistent logger name
    if output2console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Set console logging level
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
def config_log_file(log_file:str):
    logger = logging.getLogger(LOGGER_NAME)  # Use a consistent logger name
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s %(levelname)s %(message)s')
    handler = TimedRotatingFileHandler(log_file, when='midnight',backupCount=10)
    handler.setFormatter(formatter)
    i = -1
    for k,handler in enumerate(logger.handlers):
        if isinstance(handler,TimedRotatingFileHandler):
            i = k
            break
    if i!=-1:
        logger.handlers.pop(i)
    logger.addHandler(handler)
    
def get_logger():
    # Create a logger instance
    logger = logging.getLogger(LOGGER_NAME)  # Use a consistent logger name
    return logger