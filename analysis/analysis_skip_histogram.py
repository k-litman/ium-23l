import json

import pandas as pd
import matplotlib.pyplot as plt

DATA_ROOT = './data/input_data/v2/'


def calculate_skip_time_differences(data):
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    grouped = df.groupby('session_id').apply(lambda x: x.sort_values(by='timestamp')).reset_index(drop=True)

    time_diffs = []
    for session_id, group in grouped.groupby('session_id'):
        play_timestamps = group.loc[group['event_type'] == 'PLAY', ['timestamp', 'track_id']].reset_index(drop=True)
        skip_timestamps = group.loc[group['event_type'] == 'SKIP', ['timestamp', 'track_id']].reset_index(drop=True)

        for _, skip_row in skip_timestamps.iterrows():
            matching_play_rows = play_timestamps.loc[play_timestamps['track_id'] == skip_row['track_id']]
            matching_play_rows = matching_play_rows[matching_play_rows['timestamp'] < skip_row['timestamp']]

            if not matching_play_rows.empty:
                closest_play_row = matching_play_rows.loc[matching_play_rows['timestamp'].idxmax()]
                time_diff = (skip_row['timestamp'] - closest_play_row['timestamp']).total_seconds()
                time_diffs.append(time_diff)

    return time_diffs


def plot_skip_time_histogram(time_diffs):
    plt.hist(time_diffs, bins='auto', edgecolor='black')
    plt.xlabel('Czas pomijania (s)')
    plt.ylabel('Liczba pominięć')
    plt.title('Rozkład czasu pomijania utworów')
    plt.show()


if __name__ == '__main__':
    data = []
    with open(DATA_ROOT + 'sessions.jsonl', 'r') as f:
        for line in f:
            data.append(json.loads(line))

    time_diffs = calculate_skip_time_differences(data)

    if time_diffs:
        plot_skip_time_histogram(time_diffs)
