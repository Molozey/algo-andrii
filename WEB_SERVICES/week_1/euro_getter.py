import requests
import re

html = 'Курс EUR на сегодня 32.12, на завтра 33.12'
match = re.search(r'EUR\D+(\d+.\d+)', html)
print(match.group(1))
