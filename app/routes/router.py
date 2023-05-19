from fastapi import APIRouter


from app.routes.predict import router as PredictRouter


router = APIRouter()

router.include_router(PredictRouter, tags=["predict"], prefix="/predict")

