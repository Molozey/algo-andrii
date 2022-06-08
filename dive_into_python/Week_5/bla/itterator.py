class MyRangeIterator:
    def __init__(self, end):
        self.current = 0
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        current = self.current
        if self.current >= self.end:
            raise StopIteration
        else:
            self.current += 1
        return current

    def __str__(self):
        return ('ITERATOR WITH END POINT = {0}'.format(self.end))


counter = MyRangeIterator(3)
print(counter)
print('------')
for i in counter:
    print(i)
