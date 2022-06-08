import os.path
import tempfile
from pathlib import Path


class File:
    def __init__(self, path):
        xXx = '''
        if not (os.path.exists(path)) and (not os.path.join(tempfile.gettempdir(), path)):
            temp_path = os.path.join(tempfile.gettempdir(), path)
            with open(temp_path, 'w') as f:
                f.write('')
            print(temp_path, '\n creation success is:', os.path.exists(temp_path))
            path = temp_path
        '''
        if not os.path.exists(path):
            Path(path).mkdir(exist_ok=True)
            with open(path, 'w') as f:
                f.write('')
        self.path = path
        self.current = 0
        with open(path, 'r') as store:
            lines_num = len(store.readlines())
        self.lines_count = lines_num

    def __str__(self):
        return self.path

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.lines_count:
            self.current = 0
            raise StopIteration

        with open(self.path, 'r') as store:
            line = store.readlines()[self.current]
            self.current += 1
        return line

    def read(self):
        with open(self.path, 'r') as store:
            result = store.read()
        return result

    def write(self, input_text):
        with open(self.path, 'w') as store:
            store.write(input_text)

    def __add__(self, obj_2):
        obj = File(os.path.join(tempfile.gettempdir(), "tmp.txt"))
        obj.write(self.read() + obj_2.read())
        return obj

'''
    def __add__(self, obj):
        lines_array = []
        with open(self.path, 'r') as store:
            data = store.readlines()
            for _ in data:
                lines_array.append(_)
        with open(obj.path, 'r') as store:
            data = store.readlines()
            for _ in data:
                lines_array.append(_)
        splited_path = os.path.join(tempfile.gettempdir(), self.path + obj.path)
        print(splited_path)
        new_class = File(splited_path)
        string = ''
        for _ in lines_array:
            string += _
        new_class.write(string)
        return new_class
'''
x = '''
path = 'some_filename'
path_2 = 'some_filename_2'
file = File(path)
#print(file.read())
#print(file.write('Pickle Rick! \nMorty play something'))
#print(file.read())
#for line in file:
#    print(ascii(line))
file2 = File(path_2)

new_file = file + file2
print(isinstance(new_file, File))
print(type(new_file))
for line in new_file:
    print(ascii(line))
print(new_file)
'''

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
    
#'''
