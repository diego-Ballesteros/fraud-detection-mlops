"""Project configuration: filesystem paths, environment settings and fixed constants.

Every module takes its paths, credentials and reproducibility constants from the
``settings`` instance defined here instead of hardcoding them.
"""

from pathlib import Path
from typing import ClassVar, Final

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved from this file's location, not from the current working directory, so a script
# launched from any folder reads the same .env and the same data directories.
# Layout: <root>/src/fraud_detection/config.py, hence parents[2] is <root>.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Project settings.

    UPPERCASE attributes are class-level constants: read-only on the instance and never
    read from the environment. Lowercase fields are read from the process environment or
    from the ``.env`` file at the project root; all of them are optional, so importing the
    package never fails when no credentials are configured (as in CI).
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
    )

    RANDOM_STATE: ClassVar[int] = 42
    """Seed for every source of randomness in the project.

    Exact reproducibility from a clean clone is a verified requirement, so no module
    hardcodes its own seed: all of them take this one.
    """

    REFERENCE_LAST_STEP: ClassVar[int] = 400
    """Last ``step`` of the reference period, used for training and fitting.

    The temporal cut is fixed by the deliverable's assignment statement and is a hard rule
    in CLAUDE.md: no row after this step may influence the fitting of anything, whether
    the model, the preprocessor, a threshold or a statistic.
    """

    PRODUCTION_FIRST_STEP: ClassVar[int] = REFERENCE_LAST_STEP + 1
    """First ``step`` of the production period, used only for evaluation and monitoring.

    Derived from ``REFERENCE_LAST_STEP`` so the two constants cannot drift apart and leave
    a gap of steps that nobody evaluates. Same source as the reference cut: the
    deliverable's assignment statement, enforced as a hard rule in CLAUDE.md.
    """

    ROOT_DIR: ClassVar[Path] = PROJECT_ROOT
    RAW_DATA_DIR: ClassVar[Path] = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR: ClassVar[Path] = PROJECT_ROOT / "data" / "processed"
    DOCS_DIR: ClassVar[Path] = PROJECT_ROOT / "docs"
    REPORTS_DIR: ClassVar[Path] = PROJECT_ROOT / "reports"

    mlflow_tracking_uri: str | None = None
    mlflow_tracking_username: str | None = None
    mlflow_tracking_password: SecretStr | None = None


settings: Final[Settings] = Settings()
