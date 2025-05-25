from django.shortcuts import render
from .data import PROJECTS

def index(request):
    return render(request, "home/index.html", {"projects":PROJECTS})