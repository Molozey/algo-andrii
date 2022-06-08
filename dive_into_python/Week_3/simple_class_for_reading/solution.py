from pathlib import Path
import argparse


class FileReader:
    def __init__(self, path):
        storage_path = Path(path)
        self.path = storage_path

    def read(self):
        try:
            if self.path.exists():
                with open(self.path, 'r') as file:
                    return file.read()
            else:
                raise FileNotFoundError
        except FileNotFoundError:
            return ''


parser = argparse.ArgumentParser()
parser.add_argument("--path", help='Path to file')
args = parser.parse_args()
#reader = FileReader(args.path)
#text = reader.read()

