import sys
number = int(sys.argv[1])
ladder_char = '#'


def plotter(char, position, number):
    ready_string = (number - position) * ' ' + position * char
    return ready_string


for position in range(1, number+1):
    print(plotter(ladder_char, position, number))
