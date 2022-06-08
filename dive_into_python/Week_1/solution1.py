import sys

input_string = sys.argv[1]
summa = 0
for _ in input_string:
    if _.isdigit():
        summa += int(_)
    else:
        continue
print(summa)