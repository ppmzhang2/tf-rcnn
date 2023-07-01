"""Project init."""
import os

from . import config

env = os.getenv("ENV", "test")

cfg = config.Config if env == "prod" else config.TestConfig()

cfg.configure_logger(__name__)
