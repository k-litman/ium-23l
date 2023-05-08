import json
from datetime import datetime

INPUT_FILE = '../data/v3/sessions.jsonl'
OUTPUT_FILE = '../data/v3/sessions_transformed.jsonl'

with open(INPUT_FILE) as f:
    lines = f.readlines()

tracks = {}

with open(INPUT_FILE) as f:
    lines = f.readlines()

songs = {}
advertisements = {}


output = []


for line in lines:
    data = json.loads(line)


    if data['event_type'] == 'ADVERTISEMENT':
        key = (data['user_id'], data['session_id'])
        advertisements[key] = True

    elif data['event_type'] == 'PLAY':
        advertisements_key = (data['user_id'], data['session_id'])
        key = (data['user_id'], data['session_id'])
        if key in songs:
            output.append(songs[key])


        has_advertisement_before = advertisements_key in advertisements and advertisements[advertisements_key]

        songs[key] = {
            'timestamp':  data['timestamp'],
            'user_id': data['user_id'],
            'track_id': data['track_id'],
            'session_id': data['session_id'],
            'advertisementBefore': has_advertisement_before,
            'skipped': False,
            'likedWhilePlaying': False,
        }

        advertisements[advertisements_key] = False

    elif data['event_type'] == 'LIKE':
        key = (data['user_id'], data['session_id'])
        songs[key]['likedWhilePlaying'] = True

    elif data['event_type'] == 'SKIP':
        key = (data['user_id'], data['session_id'])
        songs[key]['skipped'] = True

for key, value in songs.items():
    output.append(value)



def sort_key(obj):
    ts = obj['timestamp']
    return datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.%f') if '.' in ts else datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S')

output = sorted(output, key=sort_key)


with open(OUTPUT_FILE, 'w') as f:
    for line in output:
        f.write(json.dumps(line) + '\n')