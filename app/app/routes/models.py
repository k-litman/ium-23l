from fastapi import APIRouter

from app.schema.dto.models import ModelsResponse

router = APIRouter()


@router.get("/")
async def get_models() -> ModelsResponse:
    return ModelsResponse(models=["simple", "advanced"])
