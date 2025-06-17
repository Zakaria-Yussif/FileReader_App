from django.shortcuts import render

# Create your views here.
from django.shortcuts import render

def index(request):
    message = "Hello, world!"
    return render(request, 'myFileReader/index.html', {'message': message})


def submit_message(request):
    message = request.POST.get('message')

def upload_datafiles(request):
    return render(request, 'myFileReader/upload_data.html')

def clear_message(request):
    return render(request, 'myFileReader/clear_message.html')
def clear_message_data(request):
    return render(request, 'myFileReader/clear_message_data.html')