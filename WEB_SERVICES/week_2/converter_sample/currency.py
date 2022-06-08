from bs4 import BeautifulSoup
from decimal import Decimal


def convert(amount: Decimal, cur_from: str, cur_to: str, date: str, requests):
    def logical_func(character):
        try:
            if character.parent.CharCode.text in [cur_from, cur_to]:
                return True
            else:
                return False
        except AttributeError:
            return False

    URL = f"https://www.cbr.ru/scripts/XML_daily.asp?date_req={date}"
    response = requests.get(URL)  # Использовать переданный requests
    soup = BeautifulSoup(response.content, 'xml')
    parsed = list(filter(logical_func, soup.find_all('CharCode')))
    if cur_from == 'RUR':
        OUT_VAL = list(filter(lambda x: x.parent.CharCode.text == cur_to, parsed))[0]
        #NOMINAL = OUT_VAL.parent.Nominal.text
        NOMINAL = Decimal(float(OUT_VAL.parent.Nominal.text.replace(',', '.')))
        VAL = Decimal(float(OUT_VAL.parent.Value.text.replace(',', '.')))
        ret = amount * NOMINAL / VAL

    if cur_from != 'RUR':
        NOM1 = Decimal(float(parsed[0].parent.Nominal.text.replace(',', '.')))
        NOM2 = Decimal(float(parsed[1].parent.Nominal.text.replace(',', '.')))
        VAL1 = Decimal(float(parsed[0].parent.Value.text.replace(',', '.')))
        VAL2 = Decimal(float(parsed[1].parent.Value.text.replace(',', '.')))
        ret = amount * NOM2 / NOM1 * VAL1 / VAL2
        
    return ret.quantize(Decimal('1.0000'))  # не забыть про округление до 4х знаков после запятой
