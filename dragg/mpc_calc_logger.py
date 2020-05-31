import logging
import os


class MPCCalcLogger:
    """A logger specific for the tasks of the MPCCalcLogger"""

    def __init__(self):
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
        self.name = 'mpc_calc'
        self.logger = logging.getLogger(self.name)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.fh = logging.FileHandler(f"{self.name}_logger.log")
        self.fh.setFormatter(self.formatter)
        self.logger.addHandler(self.fh)
