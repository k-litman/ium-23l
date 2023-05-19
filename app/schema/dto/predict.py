from pydantic import BaseModel


class PredictRequestDTO(BaseModel):
    track_id: str
    favourite_genres: list[str]
