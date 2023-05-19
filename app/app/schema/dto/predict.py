from pydantic import BaseModel


class PredictRequest(BaseModel):
    track_id: str
    favourite_genres: list[str]


class PredictResponse(BaseModel):
    skipped: bool
