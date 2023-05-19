import pandas as pd
import json

from matplotlib import pyplot as plt

DATA_ROOT = './data/input_data/v2/'


with open(DATA_ROOT + 'tracks.jsonl') as f:
    tracks = [json.loads(line) for line in f]

df = pd.DataFrame(tracks)
df['duration_sec'] = df['duration_ms'] / 1000

p1 = df['duration_sec'].quantile(0.01)
p99 = df['duration_sec'].quantile(0.99)

plt.hist(df['duration_sec'], bins=30, range=(0, 900))
plt.xlabel('Czas trwania (sekundy)')
plt.ylabel('Liczba utworów')
plt.title('Rozkład czasu trwania utworów')
plt.axvline(p1, color='red', linestyle='dashed', linewidth=2)
plt.axvline(p99, color='red', linestyle='dashed', linewidth=2)
plt.annotate(f'1%: {p1:.1f} s', xy=(p1+5, 10), color='red')
plt.annotate(f'99%: {p99:.1f} s', xy=(p99+5, 10), color='red')
plt.show()