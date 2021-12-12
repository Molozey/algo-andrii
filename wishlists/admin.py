from django.contrib import admin
from .models import Wishlist
# Register your models here.

class WishlistAdmin(admin.ModelAdmin):
    class Meta:
        model = Wishlist

admin.site.register(Wishlist, WishlistAdmin)