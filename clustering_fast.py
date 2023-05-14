import pickle

import jsonlines
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

from utils import map_genre

DATA_ROOT = 'data/v2/'
MODEL_ROOT = 'model2/'


def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as f:
        for line in f:
            data.append(line)
    return pd.DataFrame(data)


def apply_kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    return cluster_labels


artists_df = read_jsonl(DATA_ROOT + "artists.jsonl")
sessions_df = read_jsonl(DATA_ROOT + "sessions_transformed.jsonl")
track_storage_df = read_jsonl(DATA_ROOT + "track_storage.jsonl")
tracks_df = read_jsonl(DATA_ROOT + "tracks.jsonl")
users_df = read_jsonl(DATA_ROOT + "users.jsonl")

merged_df = sessions_df.merge(tracks_df, left_on="track_id", right_on="id", how="left")
merged_df = merged_df.merge(artists_df, left_on="id_artist", right_on="id", how="left", suffixes=("", "_artist"))
merged_df = merged_df.merge(track_storage_df, on="track_id", how="left")
merged_df = merged_df.merge(users_df, on="user_id", how="left")

used_columns = [
    'skipped',
    'genres',
    'favourite_genres',
]
to_remove_columns = list(set(merged_df.columns.tolist()) - set(used_columns))

merged_df = merged_df.drop(to_remove_columns, axis=1)

unique_genres = list(set([genre for genres in merged_df['genres'] for genre in genres])) + list(
    set([genre for genres in merged_df['favourite_genres'] for genre in genres]))

vectorizer = TfidfVectorizer()
vectorized = vectorizer.fit_transform(unique_genres)

n_components = 50
pca = PCA(n_components=n_components)
vectorized = pca.fit_transform(vectorized.toarray())

num_clusters = 200
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(vectorized)
labels = kmeans.labels_

clustered_genres = {}
for genre, label, feature in zip(unique_genres, labels, vectorized):
    if label not in clustered_genres:
        clustered_genres[label] = []
    clustered_genres[label].append((genre, feature))

genre_to_cluster = {genre: label for label, genres in clustered_genres.items() for genre, _ in genres}

print("started mapping for genres")
merged_df['genres'] = merged_df['genres'].apply(
    lambda x: [map_genre(genre, genre_to_cluster, kmeans, clustered_genres) for genre in x])
print("genres mapped")
merged_df['favourite_genres'] = merged_df['favourite_genres'].apply(
    lambda x: [map_genre(genre, genre_to_cluster, kmeans, clustered_genres) for genre in x])
print("favourite genres mapped")

print("deleting not unique genres")
merged_df['genres'] = merged_df['genres'].apply(lambda x: list(set(x)))
print("deleting not unique favourite genres")
merged_df['favourite_genres'] = merged_df['favourite_genres'].apply(lambda x: list(set(x)))

print(merged_df.head())

X = merged_df.drop("skipped", axis=1)
y = merged_df["skipped"]

mlb_genres = MultiLabelBinarizer()
mlb_favourite_genres = MultiLabelBinarizer()

X_genres = mlb_genres.fit_transform(merged_df['genres'])
X_favourite_genres = mlb_favourite_genres.fit_transform(merged_df['favourite_genres'])

# Combine the transformed columns into a single NumPy array
X = np.hstack((X_genres, X_favourite_genres))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train_scaled, y_train)

with open(MODEL_ROOT + 'X_resampled.pkl', 'wb') as f:
    pickle.dump(X_resampled, f)

with open(MODEL_ROOT + 'y_resampled.pkl', 'wb') as f:
    pickle.dump(y_resampled, f)

with open(MODEL_ROOT + 'X.pkl', 'wb') as f:
    pickle.dump(X, f)

with open(MODEL_ROOT + 'X_test_scaled.pkl', 'wb') as f:
    pickle.dump(X_test_scaled, f)

with open(MODEL_ROOT + 'y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open(MODEL_ROOT + 'y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

with open(MODEL_ROOT + 'vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open(MODEL_ROOT + 'pca.pkl', 'wb') as f:
    pickle.dump(pca, f)

with open(MODEL_ROOT + 'mlb_genres.pkl', 'wb') as f:
    pickle.dump(mlb_genres, f)

with open(MODEL_ROOT + 'mlb_favourite_genres.pkl', 'wb') as f:
    pickle.dump(mlb_favourite_genres, f)

with open(MODEL_ROOT + 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open(MODEL_ROOT + "genre_to_cluster.pkl", "wb") as f:
    pickle.dump(genre_to_cluster, f)

with open(MODEL_ROOT + "kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open(MODEL_ROOT + "clustered_genres.pkl", "wb") as f:
    pickle.dump(clustered_genres, f)

with open(MODEL_ROOT + "input_size.txt", "w") as f:
    f.write(str(X_train_scaled.shape[1]))
