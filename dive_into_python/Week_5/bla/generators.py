def MyRange(top):
    current = 0
    while current < top:
        yield current
        current += 1


for i in MyRange(4):
    print(i)
