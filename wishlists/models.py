from django.db import models
from forta.models import TickerInformation
# Create your models here.

class Wishlist(models.Model):
    tickers = models.ManyToManyField(TickerInformation, null=True, blank=True)



