import requests
from decimal import Decimal
from currency import convert


correct = Decimal('3754.8057')
#correct = Decimal('92628.4452')
result = convert(Decimal("1000.1000"), 'RUR', 'JPY', "17/02/2005", requests)
#result = convert(Decimal("1000.1000"), 'ZAR', 'KRW', "26/02/2018", requests)
if result == correct:
    print("Correct")
else:
    print("Incorrect: %s != %s" % (result, correct))
