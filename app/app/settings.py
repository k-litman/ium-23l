from functools import lru_cache
from os import getenv

from pydantic import BaseSettings


class Settings(BaseSettings):
    db_username: str = getenv("DB_USERNAME")
    db_password: str = getenv("DB_PASSWORD")
    db_host: str = getenv("DB_HOST")

    cuda_use_gpu: int = int(getenv("CUDA_USE_GPU", "1"))

    data_root: str = getenv("DATA_ROOT", "/data")
    data_version: str = getenv("DATA_VERSION", "v2")


@lru_cache()
def get_settings():
    return Settings()
