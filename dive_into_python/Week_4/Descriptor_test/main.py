from pathlib import Path
storage_path = 'storage.txt'
if not Path(storage_path).exists():
    open(storage_path, 'w').close()

class ImportantValue:
    def __init__(self, amount):
        self.amount = amount

    def __get__(self, instance, owner):
        return self.amount

    def __set__(self, instance, value):
        with open(storage_path, 'a') as f:
            f.write(str(value) + '\n')
        self.amount = value


class Account:
    amount = ImportantValue(100)


bobs_account = Account()
bobs_account.amount = 150
bobs_account.amount = 200
