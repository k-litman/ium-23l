import logging
from random import choice

from fastapi import APIRouter, HTTPException

from app.models.common import PredictionHelper
from app.schema.dto.predict import PredictRequestDTO

logger = logging.getLogger(__name__)

router = APIRouter()


# Predict with random model
@router.post('/random')
async def perform_prediction_with_random_model(request: PredictRequestDTO):
    model_name = choice(('simple', 'advanced'))

    logger.info(f"Random model: {model_name}")

    try:
        helper = PredictionHelper(model_name)
    except NotImplementedError as e:
        raise HTTPException(status_code=404, detail=str(e))

    result = await helper.predict(request)

    return {"skipped": bool(result)}


# Predict with specified model
@router.post('/{model_name}')
async def perform_prediction(model_name: str, request: PredictRequestDTO):
    try:
        helper = PredictionHelper(model_name)
    except NotImplementedError as e:
        raise HTTPException(status_code=404, detail=str(e))

    result = await helper.predict(request)

    return {"skipped": bool(result)}
