from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./legalai.db"
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_model: str = "gemma3:1b"
    embedding_model: str = "nomic-embed-text"
    upload_dir: str = "uploads"
    app_username: str = "admin"
    app_password: str = "admin"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
