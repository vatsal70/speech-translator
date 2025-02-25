from django.urls import path
from . import views
from .views import AudioTranslateView



urlpatterns = [
    path('', views.home, name='home'),
    path('translate-audio/', AudioTranslateView.as_view(), name='translate-audio'),
]