from django.shortcuts import render, redirect, reverse, HttpResponseRedirect
from .models import Wishlist
from forta.models import TickerInformation

# Create your views here.

def view(request):
    wishlists = Wishlist.objects.all()
    context = {"wishlists": wishlists}
    return render(request, 'base_wishlist.html', context)

def update_wishlist(request, pk):
    wishlist = Wishlist.objects.all()[0]
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
