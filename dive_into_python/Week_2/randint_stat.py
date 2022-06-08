import random

store = set()
while True:
    new = random.randint(1,10)
    if new not in store:
        store.add(new)
    else:
        break
print(store)
print(len(store) + 1)
