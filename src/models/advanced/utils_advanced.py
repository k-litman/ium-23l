import numpy as np
from scipy.sparse import issparse


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
