import json
from functools import wraps

def to_json(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = json.dumps(func(*args, **kwargs))
        return result
    return wrapper

@to_json
def get_data():
    return {'data': 42}

get_data()
