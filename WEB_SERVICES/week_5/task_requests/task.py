import requests
from requests.auth import HTTPBasicAuth
url = 'https://datasend.webpython.graders.eldf.ru/submissions/1/'

resp = requests.request('POST', url, headers={'Authorization': 'Basic YWxsYWRpbjpvcGVuc2VzYW1l'})
print(resp.text.encode('utf-8').decode('unicode-escape'))
url = 'https://datasend.webpython.graders.eldf.ru/submissions/secretlocation/'
resp = requests.request('PUT', url, auth=HTTPBasicAuth('alibaba', '40razboinikov'))
print(resp.text.encode('utf-8').decode('unicode-escape'))