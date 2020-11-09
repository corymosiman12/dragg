import logging
import os

progress_lvl = 25
logging.addLevelName(progress_lvl, "PROG")

def progress(self, message, *args, **kws):
    if self.isEnabledFor(progress_lvl):
        # Yes, logger takes its '*args' as 'args'.
        self._log(progress_lvl, message, args, **kws)

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
