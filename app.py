import pickle
from typing import List

import torch
import torch.nn as nn
from fastapi import FastAPI
from jsonlines import jsonlines
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer


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


MODEL_ROOT = 'model/'
# Load input_size from a file
with open(MODEL_ROOT + "input_size.txt", "r") as f:
    input_size = int(f.read())

with open(MODEL_ROOT + "mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

with open(MODEL_ROOT + "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(MODEL_ROOT + "columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MLPClassifierPytorch(input_size).to(device)
model.load_state_dict(torch.load(MODEL_ROOT + "mlp_model.pth"))
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


def process_input_data(track_id: str, favourite_genres: List[str], mlb: MultiLabelBinarizer, scaler: StandardScaler,
                       model_columns):
    # Load the required data
    artists_df = read_jsonl(DATA_ROOT + "artists.jsonl")
    track_storage_df = read_jsonl(DATA_ROOT + "track_storage.jsonl")
    tracks_df = read_jsonl(DATA_ROOT + "tracks.jsonl")

    # Get the track data for the given track_id
    track_data = tracks_df.loc[tracks_df['id'] == track_id]

    track_data.rename(columns={'id': 'track_id'}, inplace=True)

    # Merge the track data with track_storage and artists data
    track_data = track_data.merge(track_storage_df, on="track_id", how="left")
    track_data = track_data.merge(artists_df, left_on="id_artist", right_on="id", how="left", suffixes=("", "_artist"))

    # Binarize the genres
    genres_binarized = mlb.transform(track_data['genres'])
    genres_df = pd.DataFrame(genres_binarized, columns=mlb.classes_)
    genres_df.columns = "genre_" + genres_df.columns

    # Binarize the favourite_genres
    fav_genres_binarized = mlb.transform([favourite_genres])
    fav_genres_df = pd.DataFrame(fav_genres_binarized, columns="favourite_genre_" + mlb.classes_)

    # Create a DataFrame with the structure of model_columns filled with zeros
    input_data = pd.DataFrame(columns=model_columns, data=np.zeros((1, len(model_columns))))

    # Update the input_data with the information from track_data, genres_df, and fav_genres_df
    for col in genres_df.columns:
        input_data[col] = genres_df[col].values[0]

    for col in fav_genres_df.columns:
        input_data[col] = fav_genres_df[col].values[0]

    # Scale the DataFrame
    scaled_data = scaler.transform(input_data)

    return scaled_data


@app.post("/predict-skipped")
async def predict_skipped(request: PredictRequest):
    # Process the input data using the new function
    input_data = process_input_data(request.track_id, request.favourite_genres, mlb, scaler, model_columns)

    # Convert the input data to a PyTorch tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float).to(device)

    # Get the prediction using the pre-trained model
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.round(torch.sigmoid(output[0][0])).item()

    # Return the prediction as a JSON response
    return {"skipped": bool(prediction)}
