def printer(*args):
    print(type(args))
    for i, _ in enumerate(args):
        print(f'Number {i}: {_}')


printer(*[1, 2, 3, 4], *[2, 3, 4])


def dict_printer(**kwargs):
    print(type(kwargs))
    for key, value in kwargs.items():
        print('{}: {}'.format(key, value))


dict_printer(a=11, b=20)
small_dict = {'user': 24,
              'property': {
                  'HEIGHT': 62,
                  'WEIGHT': 90,
              }}
dict_printer(**small_dict)
