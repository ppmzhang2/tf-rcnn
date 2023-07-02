"""Project Configuration."""
import os
import sys
from logging.config import dictConfig

basedir = os.path.abspath(os.path.dirname(__file__))
srcdir = os.path.abspath(os.path.join(basedir, os.pardir))
rootdir = os.path.abspath(os.path.join(srcdir, os.pardir))


class Config:
    """Provide default Configuration."""

    # logging
    LOG_LEVEL = "INFO"
    LOG_LINE_FORMAT = "%(asctime)s %(levelname)-5s %(threadName)s: %(message)s"
    LOG_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"

    @classmethod
    def configure_logger(cls, root_module_name: str) -> None:
        """Configure logging."""
        dictConfig({
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "stdout_formatter": {
                    "format": cls.LOG_LINE_FORMAT,
                    "datefmt": cls.LOG_DATETIME_FORMAT,
                },
            },
            "handlers": {
                "stdout_handler": {
                    "level": cls.LOG_LEVEL,
                    "formatter": "stdout_formatter",
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                },
            },
            "loggers": {
                root_module_name: {
                    "handlers": ["stdout_handler"],
                    "level": cls.LOG_LEVEL,
                    "propagate": True,
                },
            },
        })

    # folders
    DATADIR = os.path.join(rootdir, "data")
    MODELDIR = os.path.join(rootdir, "model_config")

    # data
    EPS = 1e-4
    STRIDE = 32
    W = 512  # original image width
    H = 512  # original image height
    C = 3  # number of channels
    W_FM = W // STRIDE  # feature map width
    H_FM = H // STRIDE  # feature map height
    N_ANCHOR = 9  # number of anchors per grid cell
    MAX_BOX = 20


class TestConfig(Config):
    """Provide Testing Configuration."""
    LOG_LEVEL = "DEBUG"
