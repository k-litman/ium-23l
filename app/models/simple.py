import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

from app.models.common import PredictionHelper, AbstractMLPClassifier, tracks_df, track_storage_df, artists_df

from src.models.advanced.utils_advanced import map_genre


class SimpleMLPClassifier(AbstractMLPClassifier):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_size, 50)
        self.layer2 = torch.nn.Linear(50, 30)
        self.layer3 = torch.nn.Linear(30, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class SimplePredictionHelper(PredictionHelper):
    def __init__(self, model_name: str):
        self.model_path = model_name
        self.model_class = SimpleMLPClassifier
        self.model = None
        self.load_data()
        self.eval_model()

    def process_input(self, track_id: str, favourite_genres: list[str], mlb_genres: MultiLabelBinarizer,
                           mlb_favourite_genres: MultiLabelBinarizer, scaler: StandardScaler):
        track_data = tracks_df.loc[tracks_df['id'] == track_id]
        track_data.rename(columns={'id': 'track_id'}, inplace=True)

        track_data = track_data.merge(track_storage_df, on="track_id", how="left")
        track_data = track_data.merge(artists_df, left_on="id_artist", right_on="id", how="left",
                                      suffixes=("", "_artist"))

        new_data = pd.DataFrame(
            {'track_id': [track_id], 'genres': [track_data['genres'].values[0]],
             'favourite_genres': [favourite_genres]})

        new_data['genres'] = new_data['genres'].apply(
            lambda x: [map_genre(genre, self.genre_to_cluster, self.kmeans, self.clustered_genres) for genre in
                       x])
        new_data['favourite_genres'] = new_data['favourite_genres'].apply(
            lambda x: [map_genre(genre, self.genre_to_cluster, self.kmeans, self.clustered_genres) for genre in
                       x])

        new_data['genres'] = new_data['genres'].apply(lambda x: list(set(x)))
        new_data['favourite_genres'] = new_data['favourite_genres'].apply(lambda x: list(set(x)))

        new_X_genres = mlb_genres.transform(new_data['genres'])
        new_X_favourite_genres = mlb_favourite_genres.transform(new_data['favourite_genres'])

        new_X = np.hstack((new_X_genres, new_X_favourite_genres))

        new_X_scaled = scaler.transform(new_X)

        return new_X_scaled
