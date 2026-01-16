import enum
from pathlib import Path
from tempfile import gettempdir

from pydantic_settings import BaseSettings, SettingsConfigDict
from yarl import URL

TEMP_DIR = Path(gettempdir())


class LogLevel(enum.StrEnum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    host: str = "127.0.0.1"
    port: int = 8000
    # quantity of workers for uvicorn
    workers_count: int = 1
    # Enable uvicorn reloading
    reload: bool = False

    # Current environment
    environment: str = "dev"

    log_level: LogLevel = LogLevel.INFO
    # Variables for the database
    db_file: Path = TEMP_DIR / "db.sqlite3"
    db_echo: bool = False

    # TODO: add semantinc feature
    # Reranking configuration
    rerank_default_weights: dict[str, float] = {
        "text_match": 0.6,
        "price": 0.0,  # don't optimize for price by default
        "rating": 0.2,
        "popularity": 0.2,
    }
    mock_products_file: Path = Path(__file__).parent / "data" / "mock_products.json"

    @property
    def db_url(self) -> URL:
        """
        Assemble database URL from settings.

        :return: database URL.
        """
        return URL.build(scheme="sqlite+aiosqlite", path=f"///{self.db_file}")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RERANKER_",
        env_file_encoding="utf-8",
    )


settings = Settings()
