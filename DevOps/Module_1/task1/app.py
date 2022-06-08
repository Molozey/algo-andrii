
def solution(value):
    if value % 2 == 0:
        returned = "variable is a multiple of two"
    elif value % 3 == 0:
        returned = "variable is a multiple of three"
    elif value % 5 == 0:
        returned = "variable is a multiple of five"
    else:
        returned = "variable is not a multiple of 2, 3 and 5"
    return returned

print(solution(12))