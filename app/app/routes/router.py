from fastapi import APIRouter


from app.routes.predict import router as predict_router
from app.routes.models import router as models_router


router = APIRouter()

router.include_router(predict_router, tags=["predict"], prefix="/predict")
router.include_router(models_router, tags=["models"], prefix="/models")

