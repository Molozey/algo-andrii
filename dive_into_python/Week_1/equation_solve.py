import sys
#a = float(sys.argv[1])
#b = float(sys.argv[2])
#c = float(sys.argv[3])


def solver(a,b,c):
    DISCR = b ** 2 - 4 * (a*c)
    SQRT = DISCR ** .5
    x1 = ((-1 * b) + SQRT) / (2 * a)
    x2 = ((-1 * b) - SQRT) / (2 * a)
    return x1, x2


#x1, x2 = solver(a, b, c)
#print(str(int(x1))+'\n'+str(int(x2)))
