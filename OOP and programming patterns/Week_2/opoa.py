import time, math
class A:
    def __init__(self, value, string):
        self.__value = value
        self.string = string
    @property
    def value(self):
        return self.__value

    def __add__(self, other):
        return A(self.value + other.value)

    def __str__(self):
        return f'VALUE = {self.value}'

    def __len__(self):
        return 12

    def __repr__(self):
        return f'VALUE = {self.value}'


class B(A):
    def __init__(self, value, man):
        super().__init__(value, "norm")
        self.man = man


#print(B(10, 2).string)
while True:
    try:
        print('Time:', 2 * math.sin(time.perf_counter() / 50))
        time.sleep(0.5)
    except KeyboardInterrupt:
        print('END')
        break

