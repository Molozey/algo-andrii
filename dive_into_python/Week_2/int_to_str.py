from functools import partial


def change_args(*args):
    result = []
    for _ in args:
        result.append(int(_))
    return result


#print(change_args(*['1', '2', '3', '4']))
#print(list(map(lambda x: int(x), ['1', '2', '3'])))


def greeter(person, greet):
    return '{}, {}!'.format(person, greet)

hier = partial(greeter, 'Hi')
greet = partial(greeter, 'Hello')
print(greet('BOB'))
print(hier('SHKERK'))


def greeting(formal):
    def printer(name):
        return f'{formal}, {name}!'
    return printer


formall = greeting('Hello')
hierr = greeting('Hi')
print(formall('BOB'))
print(hierr('SHENK'))

