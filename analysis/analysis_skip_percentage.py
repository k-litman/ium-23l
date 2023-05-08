import json


DATA_ROOT = '../data/v2/'

plays = 0
skips = 0


with open(f'{DATA_ROOT}sessions.jsonl') as f:
    for line in f.readlines():
        data = json.loads(line)
        if data["event_type"] == "PLAY":
            plays += 1
        elif data["event_type"] == "SKIP":
            skips += 1

print(f"Percentage of skipped tracks in base data: {skips / plays * 100:.2f}%")
