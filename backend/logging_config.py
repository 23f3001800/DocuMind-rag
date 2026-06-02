import logging
import sys
from typing import Optional

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None
) -> None:
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


# In every module — use module-level logger
logger = logging.getLogger(__name__)  # name = "myapp.retriever"

# Use appropriate levels
logger.debug("Detailed: query=%s top_k=%d", query, top_k)   # dev only
logger.info("Retrieved %d chunks in %.3fs", len(chunks), t)  # normal ops
logger.warning("Similarity score low: %.3f for query: %s", score, q)
logger.error("LLM call failed: %s", str(e), exc_info=True)    # stack trace!