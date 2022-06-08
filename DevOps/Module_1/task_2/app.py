def simple_checker(number):
    if (number % 2 == 0) and (number != 1 or number != 2):
        logical = False

    if number % 2 != 0:
        logical = True
        for potential in range(2, number // 2):
            if number % potential == 0:
                logical = False
                break

    if number == 2:
        logical = True

    if number == 1:
        logical = False

    return logical

def solution(value):
    result = 0
    for number in range(1, value+1):
        if simple_checker(number):
            result += number
        else:
            continue
    return result


#print(solution(101))