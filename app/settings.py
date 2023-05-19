from os import getenv

from pydantic import BaseSettings


class Settings(BaseSettings):
    db_username: str = getenv("DB_USERNAME")
    db_password: str = getenv("DB_PASSWORD")
    db_host: str = getenv("DB_HOST")

    cuda_use_gpu: bool = getenv("CUDA_USE_GPU", False)

    data_root: str = getenv("DATA_ROOT", "/data")
    data_version: str = getenv("DATA_VERSION", "v3")


settings = Settings()
