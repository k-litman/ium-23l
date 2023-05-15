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

from utils_advanced import map_genre

DATA_ROOT = '../../../data/v2/'
MODEL_ROOT = '../../../model/advanced/'


def load_data(file_path):
    with jsonlines.open(file_path) as file:
        data = [line for line in file]
    return pd.DataFrame(data)


def perform_kmeans_clustering(data, number_of_clusters):
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=42)
    return kmeans.fit_predict(data)


artists = load_data(DATA_ROOT + "artists.jsonl")
sessions = load_data(DATA_ROOT + "sessions_transformed.jsonl")
track_storage = load_data(DATA_ROOT + "track_storage.jsonl")
tracks = load_data(DATA_ROOT + "tracks.jsonl")
users = load_data(DATA_ROOT + "users.jsonl")

merged_data = sessions.merge(tracks, left_on="track_id", right_on="id", how="left")
merged_data = merged_data.merge(artists, left_on="id_artist", right_on="id", how="left", suffixes=("", "_artist"))
merged_data = merged_data.merge(track_storage, on="track_id", how="left")
merged_data = merged_data.merge(users, on="user_id", how="left")

selected_columns = ['skipped', 'genres', 'favourite_genres', 'duration_ms']
unnecessary_columns = list(set(merged_data.columns.tolist()) - set(selected_columns))
merged_data = merged_data.drop(unnecessary_columns, axis=1)

all_genres = list(set([genre for genres in merged_data['genres'] for genre in genres])) + list(
    set([genre for genres in merged_data['favourite_genres'] for genre in genres]))

tfidf_vectorizer = TfidfVectorizer()
vectorized_genres = tfidf_vectorizer.fit_transform(all_genres)

pca_components = 50
pca = PCA(n_components=pca_components)
pca_genres = pca.fit_transform(vectorized_genres.toarray())

clusters_count = 200
kmeans = KMeans(n_clusters=clusters_count, random_state=42)
kmeans.fit(pca_genres)
cluster_labels = kmeans.labels_

clustered_genres = {}
for genre, label, feature in zip(all_genres, cluster_labels, pca_genres):
    if label not in clustered_genres:
        clustered_genres[label] = []
    clustered_genres[label].append((genre, feature))

genre_cluster_mapping = {genre: label for label, genres in clustered_genres.items() for genre, _ in genres}

merged_data['genres'] = merged_data['genres'].apply(
    lambda x: [map_genre(genre, genre_cluster_mapping, kmeans, clustered_genres) for genre in x])
merged_data['favourite_genres'] = merged_data['favourite_genres'].apply(
    lambda x: [map_genre(genre, genre_cluster_mapping, kmeans, clustered_genres) for genre in x])

merged_data['genres'] = merged_data['genres'].apply(lambda x: list(set(x)))
merged_data['favourite_genres'] = merged_data['favourite_genres'].apply(lambda x: list(set(x)))

features = merged_data.drop("skipped", axis=1)
labels = merged_data["skipped"]

multi_label_binarizer_genres = MultiLabelBinarizer()
multi_label_binarizer_favourite_genres = MultiLabelBinarizer()

transformed_genres = multi_label_binarizer_genres.fit_transform(merged_data['genres'])
transformed_favourite_genres = multi_label_binarizer_favourite_genres.fit_transform(merged_data['favourite_genres'])

# Combine the transformed columns into a single NumPy array
feature_matrix = np.hstack((transformed_genres, transformed_favourite_genres, merged_data[['duration_ms']]))

train_features, test_features, train_labels, test_labels = train_test_split(feature_matrix, labels, test_size=0.2,
                                                                            random_state=42)

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)

oversampler = SMOTE(random_state=42)
resampled_train_features, resampled_train_labels = oversampler.fit_resample(train_features_scaled, train_labels)

with open(MODEL_ROOT + 'resampled_train_features.pkl', 'wb') as file:
    pickle.dump(resampled_train_features, file)

with open(MODEL_ROOT + 'resampled_train_labels.pkl', 'wb') as file:
    pickle.dump(resampled_train_labels, file)

with open(MODEL_ROOT + 'feature_matrix.pkl', 'wb') as file:
    pickle.dump(feature_matrix, file)

with open(MODEL_ROOT + 'test_features_scaled.pkl', 'wb') as file:
    pickle.dump(test_features_scaled, file)

with open(MODEL_ROOT + 'train_labels.pkl', 'wb') as file:
    pickle.dump(train_labels, file)

with open(MODEL_ROOT + 'test_labels.pkl', 'wb') as file:
    pickle.dump(test_labels, file)

with open(MODEL_ROOT + "input_size.txt", "w") as file:
    file.write(str(train_features_scaled.shape[1]))

with open(MODEL_ROOT + 'multi_label_binarizer_genres.pkl', 'wb') as file:
    pickle.dump(multi_label_binarizer_genres, file)

with open(MODEL_ROOT + 'multi_label_binarizer_favourite_genres.pkl', 'wb') as file:
    pickle.dump(multi_label_binarizer_favourite_genres, file)

with open(MODEL_ROOT + 'scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open(MODEL_ROOT + 'vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)

with open(MODEL_ROOT + 'pca.pkl', 'wb') as file:
    pickle.dump(pca, file)

with open(MODEL_ROOT + "genre_to_cluster_mapping.pkl", "wb") as file:
    pickle.dump(genre_cluster_mapping, file)

with open(MODEL_ROOT + "kmeans.pkl", "wb") as file:
    pickle.dump(kmeans, file)

with open(MODEL_ROOT + "clustered_genres.pkl", "wb") as file:
    pickle.dump(clustered_genres, file)
