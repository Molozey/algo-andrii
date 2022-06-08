from django.shortcuts import render

from django.views import View

class FormLessonView(View):
    def get(self, request):
        hello = request.GET.get('hello')
        key = request.GET.get('key')
        return render(request=request, template_name='form.html', context={'hello': hello, 'key': key})

