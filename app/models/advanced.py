import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

from app.models.common import PredictionHelper, AbstractMLPClassifier, tracks_df, track_storage_df, artists_df
from src.models.advanced.utils_advanced import map_genre


class AdvancedMLPClassifier(AbstractMLPClassifier):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_size, 100)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.layer2 = torch.nn.Linear(100, 50)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.layer3 = torch.nn.Linear(50, 30)
        self.layer4 = torch.nn.Linear(30, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        return x


class AdvancedPredictionHelper(PredictionHelper):
    def __init__(self, model_name: str):
        self.model_path = model_name
        self.model_class = AdvancedMLPClassifier
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
            lambda x: [map_genre(genre, self.genre_to_cluster, self.kmeans, self.clustered_genres) for genre
                       in x])
        new_data['favourite_genres'] = new_data['favourite_genres'].apply(
            lambda x: [map_genre(genre, self.genre_to_cluster, self.kmeans, self.clustered_genres) for genre
                       in x])

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
