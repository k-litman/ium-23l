import pickle

import jsonlines
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

device = torch.device('cuda:0')

DATA_ROOT = 'data/v2/'
MODEL_ROOT = 'model/'


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
    # 'advertisementBefore',
    'skipped',
    # 'likedWhilePlaying',
    # 'id',
    # 'name_x',
    # 'popularity',
    # 'duration_ms',
    # 'explicit',
    # 'id_artist',
    # 'release_date',
    # 'danceability',
    # 'energy',
    # 'key',
    # 'mode',
    # 'loudness',
    # 'speechiness',
    # 'acousticness',
    # 'instrumentalness',
    # 'liveness',
    # 'valence',
    # 'tempo',
    # 'time_signature',

    # --- artists.jsonl
    # 'id_artist',
    # 'name_artist',
    'genres',

    # --- track_storage.jsonl
    # 'storage_class',
    # 'daily_cost',

    # --- users.jsonl
    # 'name_y',
    # 'city',
    # 'street',
    'favourite_genres',
    # 'premium_user'
]
to_remove_columns = list(set(merged_df.columns.tolist()) - set(used_columns))

merged_df = merged_df.drop(to_remove_columns, axis=1)

# ------------  GENRES TRANSFORMATION

mlb = MultiLabelBinarizer()
genres_binarized = mlb.fit_transform(merged_df['genres'])

genres_df = pd.DataFrame(genres_binarized, columns=mlb.classes_)

genres_df.columns = "genre_" + genres_df.columns

genres_df.reset_index(drop=True, inplace=True)

merged_df = merged_df.drop('genres', axis=1)

merged_df = pd.concat([merged_df, genres_df], axis=1)

# -------------- GENRES TRANSFORMATION END


# ------------  FAVGENRES TRANSFORMATION
mlb = MultiLabelBinarizer()
genres_binarized = mlb.fit_transform(merged_df['favourite_genres'])

genres_df = pd.DataFrame(genres_binarized, columns=mlb.classes_)

genres_df.columns = "favourite_genre_" + genres_df.columns

genres_df.reset_index(drop=True, inplace=True)

merged_df = merged_df.drop('favourite_genres', axis=1)
merged_df = pd.concat([merged_df, genres_df], axis=1)

# -------------- GENRES FAVTRANSFORMATION END


for column in merged_df.columns:
    if pd.api.types.is_numeric_dtype(merged_df[column]):
        merged_df[column] = pd.to_numeric(merged_df[column], errors='coerce')
        merged_df[column].fillna(merged_df[column].mean(), inplace=True)

X = merged_df.drop("skipped", axis=1)
y = merged_df["skipped"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train_scaled, y_train)


class MLPClassifierPytorch(nn.Module):
    def __init__(self):
        super(MLPClassifierPytorch, self).__init__()
        self.layer1 = nn.Linear(X.shape[1], 50)
        self.layer2 = nn.Linear(50, 30)
        self.layer3 = nn.Linear(30, 2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x


mlp = MLPClassifierPytorch().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

X_train_tensor = torch.FloatTensor(X_resampled).to(device)
y_train_tensor = torch.LongTensor(y_resampled).to(device)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = mlp(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")


# Save the trained model
torch.save(mlp.state_dict(), MODEL_ROOT + "mlp_model.pth")
print("Model saved as mlp_model.pth")

# Save the trained MultiLabelBinarizer and StandardScaler instances
with open(MODEL_ROOT + "mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)

with open(MODEL_ROOT + "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)


# Save input_size to a file
with open(MODEL_ROOT + "input_size.txt", "w") as f:
    f.write(str(X_train_scaled.shape[1]))


X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
y_test_tensor = torch.LongTensor(y_test.values).to(device)

with torch.no_grad():
    test_outputs = mlp(X_test_tensor)
    _, y_pred = torch.max(test_outputs, 1)

with open(MODEL_ROOT + "columns.pkl", "wb") as f:
    pickle.dump(X_train.columns, f)

accuracy = accuracy_score(y_test, y_pred.cpu().numpy())
print("Accuracy of the MLP Classifier with SMOTE (PyTorch): {:.2f}%".format(accuracy * 100))
