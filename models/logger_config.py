import logging

import logging

class ColoredFormatter(logging.Formatter):
    """Formatter adding color to log records."""

    COLORS = {
        'WARNING': '\033[33m',  # Yellow (You can change this if you don't want yellow)
        'INFO': '\033[34m',     # Blue
        'DEBUG': '\033[34m',    # Blue
        'CRITICAL': '\033[35m', # Magenta
        'ERROR': '\033[31m',    # Red (Changed to red)
        'RESET': '\033[0m'      # Reset color
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        message = super().format(record)
        return f'{log_color}{message}{self.COLORS["RESET"]}'

def setup_logger(name):
    """Sets up a colored logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Or your desired level

    handler = logging.StreamHandler()
    formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger