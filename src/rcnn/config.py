"""Project Configuration."""
import os
import sys
from logging.config import dictConfig

import numpy as np

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
            "disable_existing_loggers": True,  # Disable other existing loggers
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
                    # Prevent msg from being propagated to the parent logger
                    "propagate": False,
                },
            },
        })

    # folders
    DATADIR = os.path.join(rootdir, "data")
    MODELDIR = os.path.join(rootdir, "model_weights")  # save trained weights

    # training
    LR_INIT = 0.001  # Start with this learning rate
    R_DROP = 0.2  # dropout rate
    DS = "voc/2007"  # dataset name
    DS_PREFIX = "voc_2007"  # prefix for file names
    EPS = 1e-4

    # -------------------------------------------------------------------------
    # model
    # -------------------------------------------------------------------------
    STRIDE = 32
    SIZE_RESIZE = 600  # image size for resizing
    SIZE_IMG = 512  # training image size
    W = SIZE_IMG  # training image width
    H = SIZE_IMG  # training image height
    C = 3  # number of channels
    W_FM = W // STRIDE  # feature map width
    H_FM = H // STRIDE  # feature map height
    N_ANCHOR = 9  # number of anchors per grid cell
    N_SUPP_SCORE = 300  # number of boxes to keep after score suppression
    N_SUPP_NMS = 10  # number of boxes to keep after nms
    NMS_TH = 0.7  # nms threshold

    # -------------------------------------------------------------------------
    # dataset
    # -------------------------------------------------------------------------
    BUFFER_SIZE = 100  # buffer size for shuffling
    N_OBJ = 20  # ensure that each image has the same number of objects
    BATCH_SIZE_TR = 4  # TODO: error if >= 8
    BATCH_SIZE_TE = 32
    IMGNET_STD = np.array([58.393, 57.12, 57.375], dtype=np.float32)
    IMGNET_MEAN = np.array([123.68, 116.78, 103.94], dtype=np.float32)


class TestConfig(Config):
    """Provide Testing Configuration."""
    LOG_LEVEL = "DEBUG"
