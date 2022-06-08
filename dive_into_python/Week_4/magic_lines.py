class PascalList:
    def __init__(self, input_array=None):
        self.container = input_array or []

    def __getitem__(self, index):
        return self.container[index - 1]

    def __setitem__(self, index, value):
        self.container[index - 1] = value

    def __delitem__(self, index):
        del self.container[index - 1]

    def __str__(self):
        return self.container.__str__()

input_list = [1, 4, 9, 16, 25]
result = PascalList(input_list)
del result[1]
print(result)
