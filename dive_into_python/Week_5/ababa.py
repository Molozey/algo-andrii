string = 'ok\npalm.cpu 2.0 1150864247\npalm.cpu 0.5 1150864248\n\n'
string2 = 'ok\nblablabla 20.65 4\n\n'
print(string2)
_ = []
buffer = string2.split('\n')
_.append(buffer[0])
try:
    _.extend(buffer[1].split(' '))
except:
    print('err')
print(_)