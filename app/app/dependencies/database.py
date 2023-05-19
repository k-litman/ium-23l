import motor
from beanie import init_beanie

from app.schema.database.predict import Prediction
from app.settings import get_settings


MONGO_DETAILS = f"mongodb://{get_settings().db_host}:27017/ium"


async def connect():
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
    await init_beanie(database=client.get_default_database(), document_models=[Prediction])
