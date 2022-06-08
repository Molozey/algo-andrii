import argparse
import os
import tempfile
import json
from pathlib import Path

storage_path = os.path.join(tempfile.gettempdir(), 'store.data')
def save_json(dicts, path=storage_path):
    with open(storage_path, 'w') as storage:
        json.dump(dicts, storage)


def open_json(path=storage_path):
    if os.path.getsize(storage_path) != 0:
        with open(storage_path, 'r') as storage:
            data = json.load(storage)
        return data
    else:
        data = {}
        return data

def putter(storage, key, value):
    if storage.__contains__(key):
        storage[key].append(value)
    else:
        storage[key] = [value]

    return storage


def getter(storage, key):
    if storage.__contains__(key):
        if len(storage[key]) != 0:
            return ', '.join(str(_) for _ in storage[key])
        else:
            return None
    else:
        return None


parser = argparse.ArgumentParser()
parser.add_argument("--key")
parser.add_argument("--val")
args = parser.parse_args()


if not args.val:
    if Path(storage_path).exists():
        data = open_json()
        print(getter(data, args.key))
    else:
        print(None)
        with open(storage_path, 'w'):
            pass
else:
    if Path(storage_path).exists():
        data = open_json()
        new_data = putter(data, args.key, args.val)
        print(new_data)
        save_json(new_data)
    else:
        print(None)
        with open(storage_path, 'w'):
            pass
