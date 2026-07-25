import logging
from pathlib import Path


LOG_FILE = Path(__file__).resolve().parent.parent / "agent.log"


def get_app_logger() -> logging.Logger:
    logger = logging.getLogger("research_agent")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)

    return logger
