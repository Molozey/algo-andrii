import os.path
import tempfile


class File:
    def __init__(self, path):
        self._path = path
        if not os.path.exists(self._path):
            open(self._path, 'w').close()

    def __add__(self, other):
        obj = File(os.path.join(tempfile.gettempdir(), str(self._path[-5:]) + str(other._path[-5:]) + "tmp.txt"))
        obj.write(self.read() + other.read())
        return obj

    def __iter__(self):
        self._curr = 0
        with open(self._path, "r") as f:
            self._lines = f.readlines()
        return self

    def __next__(self):
        try:
            line = self._lines[self._curr]
            self._curr += 1
            return line
        except IndexError:
            raise StopIteration

    def __str__(self):
        return self._path

    def read(self):
        with open(self._path, "r") as f:
            return f.read()

    def write(self, data):
        with open(self._path, "w") as f:
            f.write(data)

'''
path_to_file = './storage/data/data1/some_filename'
print(os.path.exists(path_to_file))
file_obj = File(path_to_file)
print(os.path.exists(path_to_file))
print(file_obj)
print(file_obj.read())
file_obj.write('some text')
print(file_obj.read())
file_obj.write('other text')
print(file_obj.read())
file_obj_1 = File(path_to_file + '_1')
file_obj_2 = File(path_to_file + '_2')
file_obj_1.write('line 1 \n')
file_obj_2.write('line 2 \n')
new_file_obj = file_obj_1 + file_obj_2
print(isinstance(new_file_obj, File))
print(new_file_obj)
for line in new_file_obj:
    print(ascii(line))
new_path_to_obj = str(new_file_obj)
print(os.path.exists(new_path_to_obj))
file_obj_3 = File(new_path_to_obj)
print(file_obj_3)

print('-----')
path = 'some_filename'
file = File(path)
for line in file:
    print(ascii(line))
print('----')
for line in file:
    print(ascii(line))
'''
