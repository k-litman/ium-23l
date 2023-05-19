import jsonlines
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_ROOT = './data/input_data/v2/'


def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as f:
        for line in f:
            data.append(line)
    return pd.DataFrame(data)


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
    # 'timestamp',
    # 'user_id',
    # 'track_id',
    # 'session_id',
    'advertisementBefore',
    'skipped',
    'likedWhilePlaying',
    # 'id',
    # 'name_x',
    'popularity',
    'duration_ms',
    'explicit',
    # 'id_artist',
    # 'release_date',
    'danceability',
    'energy',
    'key',
    'mode',
    'loudness',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
    'time_signature',

    # --- artists.jsonl
    # 'id_artist',
    # 'name_artist',
    # 'genres',

    # --- track_storage.jsonl
    # 'storage_class',
    # 'daily_cost',

    # --- users.jsonl
    # 'name_y',
    # 'city',
    # 'street',
    # 'favourite_genres',
    'premium_user'
]
to_remove_columns = list(set(merged_df.columns.tolist()) - set(used_columns))

merged_df = merged_df.drop(to_remove_columns, axis=1)

# merged_df['user_id'] = encoder.fit_transform(merged_df['user_id'], merged_df['skipped'])

# merged_df = pd.get_dummies(merged_df, columns=["genres"], prefix="", prefix_sep="")

for column in merged_df.columns:
    if pd.api.types.is_numeric_dtype(merged_df[column]):
        merged_df[column] = pd.to_numeric(merged_df[column], errors='coerce')
        merged_df[column].fillna(merged_df[column].mean(), inplace=True)

X = merged_df.drop("skipped", axis=1)
y = merged_df["skipped"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


def predict_skip_probability(track_features):
    return model.predict_proba([track_features])[0][1]


track_features = X_test.iloc[0].tolist()
print("Skip probability:", predict_skip_probability(track_features))
