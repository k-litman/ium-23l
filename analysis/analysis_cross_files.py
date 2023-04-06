import pandas as pd
import json

DATA_ROOT = '../data/v2/'


def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def main():
    track_storage_file = DATA_ROOT + 'track_storage.jsonl'
    tracks_file = DATA_ROOT + 'tracks.jsonl'

    track_storage_data = read_jsonl_file(track_storage_file)
    tracks_data = read_jsonl_file(tracks_file)

    track_storage_df = pd.DataFrame(track_storage_data)
    tracks_df = pd.DataFrame(tracks_data)

    merged_df = track_storage_df.merge(tracks_df, left_on='track_id', right_on='id', how='left')

    missing_tracks = merged_df[merged_df['id'].isnull()]

    if missing_tracks.empty:
        print('All tracks_id has existing track_id in tracks.jsonl.')
    else:
        print(f'Errors count: {len(missing_tracks)}')
        print('Missing track_id:')
        for index, row in missing_tracks.iterrows():
            print(f" - {row['track_id']}")


if __name__ == '__main__':
    main()
