import requests
from datetime import datetime

def calc_age(uid):
    TODAY_YEAR = datetime.now().year
    ACCESS_TOKEN = '250d5811250d5811250d5811052574dc8f2250d250d58114459a94cddd9e61b8cc7ea04'
    GETTER_FRIENDS_URL = f"https://api.vk.com/method/friends.get?v=5.131&access_token={ACCESS_TOKEN}&user_id={uid}&fields=bdate"
    friends = requests.get(GETTER_FRIENDS_URL).json()['response']['items']

    FREQ_DICT = dict()
    for _ in friends:
        if 'bdate' in _.keys():
            date = _['bdate'].split('.')
            if len(date) == 3:
                POINTER = TODAY_YEAR - int(date[2])
                if POINTER in FREQ_DICT:
                    FREQ_DICT[POINTER] += 1
                else:
                    FREQ_DICT.update({POINTER: 1})

    return sorted(sorted(list(FREQ_DICT.items()), key=lambda x: x[0], reverse=False), key=lambda x: x[1], reverse=True)
