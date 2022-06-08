from pprint import pprint
import requests
import os


FIELDS = ["temp_max", "temp_min"]


class YandexWeatherForecast():

    URL = 'https://api.weather.yandex.ru/v1/forecast?'

    def __init__(self, key):
        self.key = key
        self.headers = {'X-Yandex-API-Key': key}

    def get_weather_week_forecasts(self, city, fields):
        data = requests.get(f'{self.URL}{city}', headers=self.headers).json()
        week_forecast = []
        for forecast in data['forecasts']:
            data = {'date': forecast['date']}
            for field in fields:
                value = forecast['parts']['day'].get(field, None)
                if value is not None:
                    data[field] = value
                week_forecast.append(data)
        return week_forecast


class CityInfo():
    def __init__(self, city, forecast_provider):
        self.city = city.lower()
        self._forecast_provider = forecast_provider

    def weather_forecast(self, fields):
        return self._forecast_provider.get_weather_week_forecasts(self.city, fields)



def _main():
    api_key = '547f6e4f-443e-4030-8bad-f3cfb2e1073d'
    weather_api = YandexWeatherForecast(api_key)
    city_name = 'Москва'
    city = CityInfo(city_name, weather_api)
    print(city.__dict__)
    pprint(city.weather_forecast(fields=FIELDS))


if __name__ == '__main__':
    _main()
