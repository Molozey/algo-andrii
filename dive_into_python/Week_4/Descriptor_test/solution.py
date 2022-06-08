class Value:


    def __init__(self):
        self.balance = None

    def __set__(self, instance, value):
        self.balance = value - value * instance.commission

    def __get__(self, instance, owner):
        return self.balance


class Account:
    amount = Value()

    def __init__(self, commission):
        self.commission = commission


new_account = Account(.1)
new_account.amount = 100
print(new_account.amount)
