import pandas as pd
import json

# For artists.jsonl similar code for other files
with open("./data/input_data/tracks.jsonl", "r") as file:
    lines = file.readlines()

artists_data = [json.loads(line) for line in lines]
artists_df = pd.DataFrame(artists_data)

null_counts = artists_df.isnull().sum()
minus_one_counts = (artists_df == -1).sum()
duplicated_counts = artists_df.duplicated(subset=["id"]).sum()

print("Null value counts:\n", null_counts)
print("\n-1 value counts:\n", minus_one_counts)
print("\nDuplicated value counts:\n", duplicated_counts)