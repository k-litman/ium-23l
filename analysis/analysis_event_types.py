import pandas as pd
import json

DATA_ROOT = './data/input_data/v2/'


def load_sessions(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data


def unique_event_types(data):
    df = pd.DataFrame(data)
    unique_event_types = df['event_type'].unique()

    return unique_event_types


def count_events(data, event_types):
    df = pd.DataFrame(data)

    event_counts = df['event_type'].value_counts()
    total_count = sum(event_counts[event_type] for event_type in event_types)

    return total_count


def check_skip_after_play(data):
    df = pd.DataFrame(data)
    grouped = df.groupby('session_id').apply(lambda x: x.sort_index()).reset_index(drop=True)
    result = True
    for session_id, group in grouped.groupby('session_id'):
        event_types = group['event_type'].values
        for i, event_type in enumerate(event_types):
            if event_type == 'SKIP':
                if 'PLAY' not in event_types[:i]:
                    result = False
                    break

    return result


def count_empty_track_id(data):
    df = pd.DataFrame(data)
    empty_track_id_count = len(df[df['track_id'] == ''])

    return empty_track_id_count


def calculate_skip_time_statistics(data):
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

    if time_diffs:
        time_diffs = pd.Series(time_diffs)
        statistics = {
            'mean': time_diffs.mean(),
            'median': time_diffs.median(),
            'min': time_diffs.min(),
            'max': time_diffs.max(),
            'std': time_diffs.std()
        }
    else:
        statistics = None

    return statistics


def main():
    file_path = DATA_ROOT + 'sessions.jsonl'
    data = load_sessions(file_path)

    # unique_types = unique_event_types(data)
    # for event_type in unique_types:
    #     print(event_type)

    # event_types = ['ADVERTISEMENT', 'BUY_PREMIUM']
    # count = count_events(data, event_types)
    # print(f'Sum of events: {count}')

    # count = count_empty_track_id(data)
    # print(f"Empty track_id: {count}")

    # is_valid = check_skip_after_play(data)
    # if is_valid:
    #     print("OK")

    stats = calculate_skip_time_statistics(data)

    if stats:
        print("When track was skipped:")
        for key, value in stats.items():
            print(f"{key}: {value:.2f} seconds")


if __name__ == '__main__':
    main()
