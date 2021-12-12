import json

from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import *
from .SQL_db_test import *
from .forms import *
from .FIFA_def import FIFA, get_similar



# Create your views here.
def main(request):
    companies = TickerInformation.objects.all()

    for company in companies:
       if company.logo_url == None:
           company.logo_url = 'https://www.dobro38.ru/upload/iblock/629/keg-20l.jpg'

    context = {'companies': companies}
    return render(request, 'base_main.html', context)

def company_info(request, pk):
     Host = "82.148.19.206"
     User = "molozey"
     Password = "Exeshnik08!"
     DataBaseName = "FinTechData"
     FIFA_columns = ['market_cap', 'earnings', 'revenue_growth', 'price_to_earnings']
     companies_info = TickerInformation.objects.filter(ticker=pk)
     companies_sector = TickerSector.objects.filter(ticker=pk)

     for company in companies_info:
         if company.logo_url == None:
             company.logo_url = 'https://www.dobro38.ru/upload/iblock/629/keg-20l.jpg'

     similar = list(get_similar(ticker=pk, Host=Host, Password=Password, User=User, DataBaseName=DataBaseName))
     cmp_similar = list()
     for _ in similar:
         rows = TickerInformation.objects.filter(ticker=_)
         for row in rows:
             cmp_similar.append(row)

     fifa_data_key_value = FIFA(ticker_name=pk, columns=FIFA_columns)
     fifa_data_value = list()
     fifa_data_key = list()
     for fifa_info in list(fifa_data_key_value.items())[1:]:
         fifa_data_value.append(fifa_info[1])
         fifa_data_key.append(fifa_info[0])

     context = {'companies_info': companies_info, 'companies_sector': companies_sector,
                'fifa_data_value': fifa_data_value,
                'fifa_data_key': fifa_data_key,
                'similar_companies': cmp_similar,}
     return render(request, 'base_comp_info.html', context)

