import pickle
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI
from jsonlines import jsonlines
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

from src.models.simple.simple_utils import map_genre


class MLPClassifierPytorch(nn.Module):
    def __init__(self, input_size):
        super(MLPClassifierPytorch, self).__init__()
        self.layer1 = nn.Linear(input_size, 50)
        self.layer2 = nn.Linear(50, 30)
        self.layer3 = nn.Linear(30, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


MODEL_SIMPLE_ROOT = 'model/simple/'

with open(MODEL_SIMPLE_ROOT + "input_size.txt", "r") as f:
    input_size = int(f.read())

with open(MODEL_SIMPLE_ROOT + "multi_label_binarizer_genres.pkl", "rb") as f:
    mlb_genres = pickle.load(f)

with open(MODEL_SIMPLE_ROOT + "multi_label_binarizer_favourite_genres.pkl", "rb") as f:
    mlb_favourite_genres = pickle.load(f)

with open(MODEL_SIMPLE_ROOT + "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(MODEL_SIMPLE_ROOT + "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(MODEL_SIMPLE_ROOT + "genre_to_cluster_mapping.pkl", "rb") as f:
    genre_to_cluster = pickle.load(f)

with open(MODEL_SIMPLE_ROOT + "kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open(MODEL_SIMPLE_ROOT + "clustered_genres.pkl", "rb") as f:
    clustered_genres = pickle.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MLPClassifierPytorch(input_size).to(device)
model.load_state_dict(torch.load(MODEL_SIMPLE_ROOT + "mlp_model.pth"))
model.eval()

app = FastAPI()

DATA_ROOT = 'data/v2/'


def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as f:
        for line in f:
            data.append(line)
    return pd.DataFrame(data)


class PredictRequest(BaseModel):
    track_id: str
    favourite_genres: List[str]


artists_df = read_jsonl(DATA_ROOT + "artists.jsonl")
track_storage_df = read_jsonl(DATA_ROOT + "track_storage.jsonl")
tracks_df = read_jsonl(DATA_ROOT + "tracks.jsonl")


def process_input_data(track_id: str, favourite_genres: List[str], mlb_genres: MultiLabelBinarizer,
                       mlb_favourite_genres: MultiLabelBinarizer, scaler: StandardScaler):
    track_data = tracks_df.loc[tracks_df['id'] == track_id]
    track_data.rename(columns={'id': 'track_id'}, inplace=True)

    track_data = track_data.merge(track_storage_df, on="track_id", how="left")
    track_data = track_data.merge(artists_df, left_on="id_artist", right_on="id", how="left", suffixes=("", "_artist"))

    new_data = pd.DataFrame(
        {'track_id': [track_id], 'genres': [track_data['genres'].values[0]], 'favourite_genres': [favourite_genres]})

    new_data['genres'] = new_data['genres'].apply(
        lambda x: [map_genre(genre, genre_to_cluster, kmeans, clustered_genres) for genre in x])
    new_data['favourite_genres'] = new_data['favourite_genres'].apply(
        lambda x: [map_genre(genre, genre_to_cluster, kmeans, clustered_genres) for genre in x])

    new_data['genres'] = new_data['genres'].apply(lambda x: list(set(x)))
    new_data['favourite_genres'] = new_data['favourite_genres'].apply(lambda x: list(set(x)))

    new_X_genres = mlb_genres.transform(new_data['genres'])
    new_X_favourite_genres = mlb_favourite_genres.transform(new_data['favourite_genres'])

    new_X = np.hstack((new_X_genres, new_X_favourite_genres))

    new_X_scaled = scaler.transform(new_X)

    return new_X_scaled


@app.post("/predict-skipped")
async def predict_skipped(request: PredictRequest):
    input_data = process_input_data(request.track_id, request.favourite_genres, mlb_genres, mlb_favourite_genres,
                                    scaler)
    input_tensor = torch.tensor(input_data, dtype=torch.float).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.round(torch.sigmoid(output[0][0])).item()

    return {"skipped": bool(prediction)}
