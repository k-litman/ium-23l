import pandas as pd
import matplotlib.pyplot as plt
import json

DATA_ROOT = '../data/v2/'

with open(DATA_ROOT + 'tracks.jsonl') as f:
    tracks = [json.loads(line) for line in f]

df = pd.DataFrame(tracks)

attributes = ["popularity", "explicit", "danceability", "energy", "key",
              "loudness", "speechiness", "acousticness", "instrumentalness", "liveness",
              "valence", "tempo", "time_signature"]

for attribute in attributes:
    if attribute in df.columns:
        fig, ax = plt.subplots()
        data = df[attribute]
        p1 = data.quantile(0.01)
        p99 = data.quantile(0.99)
        ax.hist(data, bins=30, range=(data.min(), data.max()))
        ax.set_xlabel(attribute.capitalize())
        ax.set_ylabel('Liczba utworów')
        ax.set_title(attribute.capitalize())
        ax.axvline(p1, color='red', linestyle='dashed', linewidth=2)
        ax.axvline(p99, color='red', linestyle='dashed', linewidth=2)
        ax.annotate(f'1%: {p1:.1f}', xy=(p1+5, 10), color='red')
        ax.annotate(f'99%: {p99:.1f}', xy=(p99+5, 10), color='red')
        plt.savefig(f'{attribute}.png', dpi=300, bbox_inches='tight')
        plt.close()
