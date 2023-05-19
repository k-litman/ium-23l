import motor
from beanie import init_beanie

from app.settings import settings


MONGO_DETAILS = f"mongodb://{settings.db_username}:{settings.db_password}@{settings.db_host}:27017"


async def connect():
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
    await init_beanie(database=client.get_default_database(), document_models=[])
