import requests

url = "https://github-not-found.com"
try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
except requests.Timeout:
    print(f'Не удалось получить ответ за указанное время по адресу {url}')
except requests.HTTPError as err:
    code = err.response.status_code
    print('Ошибка при url:{0}, code:{1}'.format(url, code))
except requests.RequestException:
    print('Ошибка скачивания url: {0}'.format(url))
else:
    print(response.content)