from pydantic import computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    VERSION: str = "1.0.0"

    # Database
    MYSQL_SERVER: str = "localhost"
    MYSQL_USER: str = "smartrent"
    MYSQL_PASSWORD: str = "password"
    MYSQL_DB: str = "smartrent_ai"
    MYSQL_PORT: int = 3306

    @computed_field  # type: ignore[misc]
    @property
    def DATABASE_URL(self) -> str:
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@"
            f"{self.MYSQL_SERVER}:{self.MYSQL_PORT}/{self.MYSQL_DB}"
        )

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # AI Configuration
    GEMINI_API_KEY: str = ""

    model_config = {"env_file": ".env", "case_sensitive": True}


settings = Settings()
