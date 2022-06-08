import requests
from datetime import datetime


TODAY_YEAR = datetime.now().year
ACCESS_TOKEN = '250d5811250d5811250d5811052574dc8f2250d250d58114459a94cddd9e61b8cc7ea04'
USERNAME = 'uncle_molozey'
GETTER_USER_URL = f'https://api.vk.com/method/users.get?v=5.131&access_token={ACCESS_TOKEN}&user_ids={USERNAME}'
USER_ID = requests.get(GETTER_USER_URL).json()['response'][0]['id']
GETTER_FRIENDS_URL = f"https://api.vk.com/method/friends.get?v=5.131&access_token={ACCESS_TOKEN}&user_id={USER_ID}&fields=bdate"
friends = requests.get(GETTER_FRIENDS_URL).json()['response']['items']



FREQ_DICT = dict({20: 1})
for _ in friends:
    if 'bdate' in _.keys():
        date = _['bdate'].split('.')
        if len(date) == 3:
            POINTER = TODAY_YEAR - int(date[2])
            if POINTER in FREQ_DICT:
                FREQ_DICT[POINTER] += 1
            else:
                FREQ_DICT.update({POINTER: 1})


print(sorted(list(FREQ_DICT.items()), key=lambda x: x[1], reverse=True))