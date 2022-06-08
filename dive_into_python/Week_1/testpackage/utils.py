def summary(a:str,b:str):
    if a.isdigit() and b.isdigit():
        return float(a) + float(b)
    else:
        print('Введите числа!')
def division(a:str,b:str):
    if a.isdigit() and b.isdigit():
        if b != '0':
            return float(a) / float(b)
        else:
            print('Деление на 0')
    else:
        print('Введите числа')
