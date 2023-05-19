from pydantic import BaseModel


class ModelsResponse(BaseModel):
    models: list[str]
