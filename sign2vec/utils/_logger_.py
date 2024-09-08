import logging

class Logger:

    def __init__(self, name, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.addHandler(logging.StreamHandler())

        self.set_format(format)

    # Add custom formatting
    def set_format(self, format):
        formatter = logging.Formatter(format)
        for handler in self.logger.handlers:
            handler.setFormatter(formatter)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)
    
    def warning(self, message):
        self.logger.warning(message)

    def debug(self, message):
        self.logger.debug(message)

    def critical(self, message):
        self.logger.critical(message)

    def exception(self, message):
        self.logger.exception(message)

    def log(self, level, message):
        self.logger.log(level, message)