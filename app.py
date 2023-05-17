import pickle
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from jsonlines import jsonlines
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

from src.models.simple.utils_simple import map_genre

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MLPClassifierSimplePytorch(nn.Module):
    def __init__(self, input_size):
        super(MLPClassifierSimplePytorch, self).__init__()
        self.layer1 = nn.Linear(input_size, 50)
        self.layer2 = nn.Linear(50, 30)
        self.layer3 = nn.Linear(30, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class MLPClassifierAdvancedPytorch(nn.Module):
    def __init__(self, input_size):
        super(MLPClassifierAdvancedPytorch, self).__init__()
        self.layer1 = nn.Linear(input_size, 100)
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(100, 50)
        self.dropout2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(50, 30)
        self.layer4 = nn.Linear(30, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        return x


MODEL_SIMPLE_ROOT = 'model/simple/'
with open(MODEL_SIMPLE_ROOT + "input_size.txt", "r") as f:
    simple_input_size = int(f.read())

with open(MODEL_SIMPLE_ROOT + "multi_label_binarizer_genres.pkl", "rb") as f:
    simple_mlb_genres = pickle.load(f)

with open(MODEL_SIMPLE_ROOT + "multi_label_binarizer_favourite_genres.pkl", "rb") as f:
    simple_mlb_favourite_genres = pickle.load(f)

with open(MODEL_SIMPLE_ROOT + "scaler.pkl", "rb") as f:
    simple_scaler = pickle.load(f)

with open(MODEL_SIMPLE_ROOT + "vectorizer.pkl", "rb") as f:
    simple_vectorizer = pickle.load(f)

with open(MODEL_SIMPLE_ROOT + "genre_to_cluster_mapping.pkl", "rb") as f:
    simple_genre_to_cluster = pickle.load(f)

with open(MODEL_SIMPLE_ROOT + "kmeans.pkl", "rb") as f:
    simple_kmeans = pickle.load(f)

with open(MODEL_SIMPLE_ROOT + "clustered_genres.pkl", "rb") as f:
    simple_clustered_genres = pickle.load(f)

simple_model = MLPClassifierSimplePytorch(simple_input_size).to(device)
simple_model.load_state_dict(torch.load(MODEL_SIMPLE_ROOT + "mlp_model.pth"))
simple_model.eval()

MODEL_ADVANCED_ROOT = 'model/advanced/'
with open(MODEL_ADVANCED_ROOT + "input_size.txt", "r") as f:
    advanced_input_size = int(f.read())

with open(MODEL_ADVANCED_ROOT + "multi_label_binarizer_genres.pkl", "rb") as f:
    advanced_mlb_genres = pickle.load(f)

with open(MODEL_ADVANCED_ROOT + "multi_label_binarizer_favourite_genres.pkl", "rb") as f:
    advanced_mlb_favourite_genres = pickle.load(f)

with open(MODEL_ADVANCED_ROOT + "scaler.pkl", "rb") as f:
    advanced_scaler = pickle.load(f)

with open(MODEL_ADVANCED_ROOT + "vectorizer.pkl", "rb") as f:
    advanced_vectorizer = pickle.load(f)

with open(MODEL_ADVANCED_ROOT + "genre_to_cluster_mapping.pkl", "rb") as f:
    advanced_genre_to_cluster = pickle.load(f)

with open(MODEL_ADVANCED_ROOT + "kmeans.pkl", "rb") as f:
    advanced_kmeans = pickle.load(f)

with open(MODEL_ADVANCED_ROOT + "clustered_genres.pkl", "rb") as f:
    advanced_clustered_genres = pickle.load(f)

advanced_model = MLPClassifierAdvancedPytorch(advanced_input_size).to(device)
advanced_model.load_state_dict(torch.load(MODEL_ADVANCED_ROOT + "mlp_model.pth"))
advanced_model.eval()

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


def process_input_simple(track_id: str, favourite_genres: List[str], mlb_genres: MultiLabelBinarizer,
                         mlb_favourite_genres: MultiLabelBinarizer, scaler: StandardScaler):
    track_data = tracks_df.loc[tracks_df['id'] == track_id]
    track_data.rename(columns={'id': 'track_id'}, inplace=True)

    track_data = track_data.merge(track_storage_df, on="track_id", how="left")
    track_data = track_data.merge(artists_df, left_on="id_artist", right_on="id", how="left", suffixes=("", "_artist"))

    new_data = pd.DataFrame(
        {'track_id': [track_id], 'genres': [track_data['genres'].values[0]], 'favourite_genres': [favourite_genres]})

    new_data['genres'] = new_data['genres'].apply(
        lambda x: [map_genre(genre, simple_genre_to_cluster, simple_kmeans, simple_clustered_genres) for genre in x])
    new_data['favourite_genres'] = new_data['favourite_genres'].apply(
        lambda x: [map_genre(genre, simple_genre_to_cluster, simple_kmeans, simple_clustered_genres) for genre in x])

    new_data['genres'] = new_data['genres'].apply(lambda x: list(set(x)))
    new_data['favourite_genres'] = new_data['favourite_genres'].apply(lambda x: list(set(x)))

    new_X_genres = mlb_genres.transform(new_data['genres'])
    new_X_favourite_genres = mlb_favourite_genres.transform(new_data['favourite_genres'])

    new_X = np.hstack((new_X_genres, new_X_favourite_genres))

    new_X_scaled = scaler.transform(new_X)

    return new_X_scaled


def process_input_advanced(track_id: str, favourite_genres: List[str], mlb_genres: MultiLabelBinarizer,
                           mlb_favourite_genres: MultiLabelBinarizer, scaler: StandardScaler):
    track_data = tracks_df.loc[tracks_df['id'] == track_id]
    track_data.rename(columns={'id': 'track_id'}, inplace=True)

    track_data = track_data.merge(track_storage_df, on="track_id", how="left")
    track_data = track_data.merge(artists_df, left_on="id_artist", right_on="id", how="left", suffixes=("", "_artist"))

    new_data = pd.DataFrame(
        {'track_id': [track_id], 'genres': [track_data['genres'].values[0]], 'favourite_genres': [favourite_genres],
         'duration_ms': [track_data['duration_ms'].values[0]],
            'popularity': [track_data['popularity'].values[0]],
            'explicit': [track_data['explicit'].values[0]],
            'danceability': [track_data['danceability'].values[0]],
            'energy': [track_data['energy'].values[0]],
            'key': [track_data['key'].values[0]],
            'loudness': [track_data['loudness'].values[0]],
            'speechiness': [track_data['speechiness'].values[0]],
            'acousticness': [track_data['acousticness'].values[0]],
            'instrumentalness': [track_data['instrumentalness'].values[0]],
            'liveness': [track_data['liveness'].values[0]],
            'valence': [track_data['valence'].values[0]],
            'tempo': [track_data['tempo'].values[0]]
         })

    new_data['genres'] = new_data['genres'].apply(
        lambda x: [map_genre(genre, advanced_genre_to_cluster, advanced_kmeans, advanced_clustered_genres) for genre in x])
    new_data['favourite_genres'] = new_data['favourite_genres'].apply(
        lambda x: [map_genre(genre, advanced_genre_to_cluster, advanced_kmeans, advanced_clustered_genres) for genre in x])

    new_data['genres'] = new_data['genres'].apply(lambda x: list(set(x)))
    new_data['favourite_genres'] = new_data['favourite_genres'].apply(lambda x: list(set(x)))

    new_X_genres = mlb_genres.transform(new_data['genres'])
    new_X_favourite_genres = mlb_favourite_genres.transform(new_data['favourite_genres'])

    new_X = np.hstack((
        new_X_genres,
        new_X_favourite_genres,
        new_data[['duration_ms']],
        new_data[['popularity']],
        new_data[['explicit']],
        new_data[['danceability']],
        new_data[['energy']],
        new_data[['key']],
        new_data[['loudness']],
        new_data[['speechiness']],
        new_data[['acousticness']],
        new_data[['instrumentalness']],
        new_data[['liveness']],
        new_data[['valence']],
        new_data[['tempo']]
    ))

    new_X_scaled = scaler.transform(new_X)

    return new_X_scaled


@app.post("/models/{model_name}/predict")
async def predict_skipped_with_specified_model(model_name, request: PredictRequest):
    if model_name == "simple":
        return predict_simple(request)
    elif model_name == "advanced":
        return predict_advanced(request)
    else:
        raise HTTPException(status_code=404, detail="Model not found")


@app.post("/models/predict")
async def predict_skipped_with_specified_model(request: PredictRequest):
    if random.randrange(0, 2):
        print("predicting using simple model")
        return predict_simple(request)
    else:
        print("predicting using advanced model")
        return predict_advanced(request)


def predict_simple(request):
    input_data = process_input_simple(request.track_id, request.favourite_genres, simple_mlb_genres,
                                      simple_mlb_favourite_genres,
                                      simple_scaler)
    input_tensor = torch.tensor(input_data, dtype=torch.float).to(device)

    with torch.no_grad():
        output = simple_model(input_tensor)
        prediction = torch.round(torch.sigmoid(output[0][0])).item()

    return {"skipped": bool(prediction)}


def predict_advanced(request):
    input_data = process_input_advanced(request.track_id, request.favourite_genres, advanced_mlb_genres,
                                        advanced_mlb_favourite_genres,
                                        advanced_scaler)
    input_tensor = torch.tensor(input_data, dtype=torch.float).to(device)

    with torch.no_grad():
        output = advanced_model(input_tensor)
        prediction = torch.round(torch.sigmoid(output[0][0])).item()

    return {"skipped": bool(prediction)}
