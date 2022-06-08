from django.shortcuts import render

# Create your views here.


def start_page(request):
    return render(request, 'application/start_page.html')

def register_page(request):
    return render(request, 'register_page.html')

def log_in_page(request):
    return render(request, 'log_in_page.html')

def topic1(request):
    return render(request, 'application/topic.html')
