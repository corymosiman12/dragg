import logging
import os


class Logger:
    """A logger for simulation outputs"""

    def __init__(self, name):
        self.name = name

        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
        self.logger = logging.getLogger(self.name)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.fh = logging.FileHandler(f"{self.name}_logger.log")
        self.fh.setFormatter(self.formatter)
        self.logger.addHandler(self.fh)
