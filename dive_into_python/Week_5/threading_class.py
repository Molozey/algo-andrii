from threading import Thread


class PrinterName(Thread):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def run(self):
        print('My name is {0}'.format(self.name))

th = PrinterName('Mike')
th.start()
th.join()
