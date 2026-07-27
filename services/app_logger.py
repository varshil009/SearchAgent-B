import logging
from pathlib import Path


LOG_FILE = Path(__file__).resolve().parent.parent / "backend.log.txt"


def get_app_logger(name: str | None = None) -> logging.Logger:
    """Return an application logger that writes to the backend-root text log."""
    logger = logging.getLogger("research_agent")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        # FileHandler flushes after every record, so the file updates while the
        # backend is running and can be tailed from the editor.
        handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)

    return logger if name is None else logger.getChild(name)
