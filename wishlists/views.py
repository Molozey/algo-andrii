from django.shortcuts import render, redirect, reverse, HttpResponseRedirect
from .models import Wishlist
from forta.models import TickerInformation
from wishlists.functions.marko import MARKO
# Create your views here.

def view(request):
    try:
        the_id = request.session['wishlist_id']
    except:
        the_id = None
    if the_id:
        wishlist = Wishlist.objects.get(id=the_id)
        CMPY_LIST = list()
        for company in wishlist.tickers.all():
            CMPY_LIST.append(company.ticker)

            STATUS, MAR = MARKO(CMPY_LIST)
            if STATUS == '+':
                context = {"wishlist": wishlist,
                           'marko': MAR}

            if STATUS == '-':
                context = {"wishlist": wishlist,
                           "marco_un": True}
    else:
        empty_message = "Looks so empty..."
        context = {"empty": True, "empty_message": empty_message}
    return render(request, 'base_wishlist.html', context)

def update_wishlist(request, pk):
    ###################################### здесь автоматом удаляется сессия после 5 мин
    request.session.set_expiry(300)
    try:
        the_id = request.session['wishlist_id']
    except:
        new_wishlist = Wishlist()
        new_wishlist.save()
        request.session['wishlist_id'] = new_wishlist.id
        the_id = new_wishlist.id

    company = None
    wishlist = Wishlist.objects.get(id=the_id)

    try:
        company = TickerInformation.objects.get(ticker=pk)
    except TickerInformation.DoesNotExist:
        pass
    except:
        pass
    if not company in wishlist.tickers.all():
        wishlist.tickers.add(company)
    else:
        wishlist.tickers.remove(company)
    return HttpResponseRedirect(reverse("wishlist"))
