import random


def logger(filename):
    def decorator(func):
        import time

        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f'Result time: {end_time - start_time}')
            with open(filename, 'w') as store:
                store.write(str(result) + '\n' + f'Time:{end_time - start_time}')
            return start_time, end_time, result
        return wrapper
    return decorator

@logger('log2.txt')
def summator(num_list):
    return sum(num_list)


list_num = [random.randint(1, 100) for _ in range(1000)]
print(summator(list_num))
