"""task_requests URL Configuration

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
from application.views import start_page, register_page, log_in_page, topic1

urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^start_page/$', start_page, name="start_page"),
    url(r'^register_page/$', register_page, name='register_page'),
    url(r'^log_in_page/$', log_in_page, name='log_in_page'),
    url(r'^topic', topic1, name='topic')
]
