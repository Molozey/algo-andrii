class File:
    def __init__(self, path):
        self.path = path
        self.current = 0
        with open(path, 'r') as store:
            lines_num = len(store.readlines())
        self.lines_count = lines_num

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.lines_count:
            raise StopIteration
        with open(self.path, 'r') as store:
            line = store.readlines()[self.current]
            self.current += 1
            return line

    def __str__(self):
        return self.path

file = File('some_filename')

for lines in file:
    print(ascii(lines))
