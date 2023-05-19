from datetime import datetime
import pickle
from typing import Any, Type

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

from abc import ABC, abstractmethod

from jsonlines import jsonlines
import logging

from app.database.operations import add_prediction
from app.schema.database.predict import Prediction
from app.schema.dto.predict import PredictRequest
from app.settings import get_settings


settings = get_settings()


logger = logging.getLogger(__name__)


def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as f:
        for line in f:
            data.append(line)
    return pd.DataFrame(data)


def get_device():
    if settings.cuda_use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            logger.warning("CUDA on GPU is not available, using CPU as fallback")
            return torch.device("cpu")
    else:
        return torch.device("cpu")


artists_df = read_jsonl(f"{settings.data_root}/input_data/{settings.data_version}/artists.jsonl")
track_storage_df = read_jsonl(f"{settings.data_root}/input_data/{settings.data_version}/track_storage.jsonl")
tracks_df = read_jsonl(f"{settings.data_root}/input_data/{settings.data_version}/tracks.jsonl")


def get_representative_genre(cluster, kmeans, clustered_genres):
    centroid = kmeans.cluster_centers_[cluster]
    points = np.array([point for _, point in clustered_genres[cluster]])

    if issparse(points[0]):
        points = np.array([point.toarray()[0] for point in points])

    distances = np.linalg.norm(points - centroid, axis=1)
    min_index = np.argmin(distances)

    return clustered_genres[cluster][min_index][0]


def map_genre(genre, genre_to_cluster, kmeans, clustered_genres):
    cluster_label = genre_to_cluster[genre]
    representative_genre = get_representative_genre(cluster_label, kmeans, clustered_genres)
    return representative_genre


class AbstractMLPClassifier(ABC, torch.nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError


class PredictionHelper(ABC):
    model_path: str
    input_size: int
    mlb_genres: MultiLabelBinarizer
    mlb_favourite_genres: MultiLabelBinarizer
    scaler: StandardScaler
    vectorizer: Any
    genre_to_cluster: Any
    kmeans: Any
    clustered_genres: Any

    model_class: Type[AbstractMLPClassifier]
    model: Any

    def __new__(cls, model_name: str):
        if model_name == 'simple':
            from app.models.simple import SimplePredictionHelper
            return super().__new__(SimplePredictionHelper)
        elif model_name == 'advanced':
            from app.models.advanced import AdvancedPredictionHelper
            return super().__new__(AdvancedPredictionHelper)
        else:
            raise NotImplementedError(f"Unknown model name: {model_name}")

    @abstractmethod
    def __init__(self, model_name: str):
        pass

    def load_data(self):
        path = f"{settings.data_root}/models/{self.model_path}"

        with open(f"{path}/input_size.txt", "r") as f:
            self.input_size = int(f.read())

        with open(f"{path}/multi_label_binarizer_genres.pkl", "rb") as f:
            self.mlb_genres = pickle.load(f)

        with open(f"{path}/multi_label_binarizer_favourite_genres.pkl", "rb") as f:
            self.mlb_favourite_genres = pickle.load(f)

        with open(f"{path}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open(f"{path}/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

        with open(f"{path}/genre_to_cluster_mapping.pkl", "rb") as f:
            self.genre_to_cluster = pickle.load(f)

        with open(f"{path}/kmeans.pkl", "rb") as f:
            self.kmeans = pickle.load(f)

        with open(f"{path}/clustered_genres.pkl", "rb") as f:
            self.clustered_genres = pickle.load(f)

    def eval_model(self):
        self.model = self.model_class(self.input_size).to(get_device())
        self.model.load_state_dict(torch.load(f"{settings.data_root}/models/{self.model_path}/mlp_model.pth"))
        self.model.eval()

    @abstractmethod
    def process_input(self, track_id: str, favourite_genres: list[str], mlb_genres: MultiLabelBinarizer,
                           mlb_favourite_genres: MultiLabelBinarizer, scaler: StandardScaler):
        raise NotImplementedError

    async def predict(self, request: PredictRequest):
        input_data = self.process_input(request.track_id, request.favourite_genres, self.mlb_genres, self.mlb_favourite_genres, self.scaler)
        tensor = torch.tensor(input_data, dtype=torch.float).to(get_device())

        with torch.no_grad():
            output = self.model(tensor)
            prediction = torch.round(torch.sigmoid(output[0][0])).item()

        return prediction

    async def save_prediction(self, request: PredictRequest, result):
        prediction = Prediction(
            model=self.model_path,
            track_id=request.track_id,
            favourite_genres=request.favourite_genres,
            result=bool(result),
            when=datetime.now()
        )

        await add_prediction(prediction)



