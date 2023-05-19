from app.schema.database.predict import Prediction


async def add_prediction(prediction: Prediction) -> Prediction:
    return await prediction.create()