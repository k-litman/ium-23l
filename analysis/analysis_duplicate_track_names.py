import json


DATA_ROOT = '../data/v2/'

duplicates = dict()


with open(f'{DATA_ROOT}tracks.jsonl') as f:
    for line in f.readlines():
        data = json.loads(line)
        if (data["id_artist"], data["name"]) in duplicates.keys():
            duplicates[(data["id_artist"], data["name"])] += 1
        else:
            duplicates[(data["id_artist"], data["name"])] = 1

for (artist, name), value in sorted(duplicates.items(), key=lambda x: -x[1]):
    if value == 1:
        break
    print(f"Duplicate entries for artist id: {artist} and track name: {name} - {value} occurrences")
