class SimpleIterator:
    def __init__(self, start, end):
        self.start = start
        self.current = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        result = self.current ** 2
        self.current += 1
        return result


for num in SimpleIterator(1,5):
    print(num)