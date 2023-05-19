from datetime import datetime
from beanie import Document


class Prediction(Document):
    model: str
    track_id: str
    favourite_genres: list[str]
    result: str
    when: datetime
