"""simple URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url

from simple_application.views import index, topic_details, return_to_main

urlpatterns = [
    path('index/', index, name='index'),
    path('admin/', admin.site.urls),
    url(r'topic/(?P<pk>\d+)/$', topic_details, name="topic_details"),
    url(r'index/', return_to_main, name='get_back')
]
