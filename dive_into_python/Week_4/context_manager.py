from time import time, sleep


class suppress_exception:
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type == self.exc_type:
            print('Nothing Happens')
            return True


with suppress_exception(ZeroDivisionError):
    num = 1/ 0


class time_manager:
    def __init__(self):
        self.start = time()
        self.end = None
        self.current = None

    def current_time(self):
        self.current = time() - self.start
        return f'Time : {self.current}'

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end = time()
        print('Total time:', self.end - self.start)
        return True

with time_manager() as timer:
    sleep(1)
    print(timer.current_time())
    sleep(1)
    print(timer.current_time())
    pass