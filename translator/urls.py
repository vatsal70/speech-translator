from django.urls import path
from . import views
from .views import AudioTranslateView1, AudioTranslateView2, AudioTranslateView



urlpatterns = [
    path('', views.home, name='home'),
    # path('translate-audio1/', AudioTranslateView1.as_view(), name='translate-audio1'),
    # path('translate-audio2/', AudioTranslateView2.as_view(), name='translate-audio2'),
    path('translate-audio/', AudioTranslateView.as_view(), name='translate-audio'),
]