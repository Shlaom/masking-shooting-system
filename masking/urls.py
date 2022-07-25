from django.urls import path

from . import views

app_name = 'masking'

urlpatterns = [
    path('', views.main, name='main'),

    path('registration', views.registration, name='registration'),

    path('face_capture', views.face_capture, name='face_capture'),

    path('video', views.video, name='video'),

    path('home', views.home, name='home'),

    # 마이페이지
    path('mypage', views.mypage, name='mypage'),

    path('masking_on', views.masking_on, name='masking_on'),

    path('masking_off', views.masking_off, name='masking_off'),

    path('mode_mosaic', views.mode_mosaic, name='mode_mosaic'),



    path('mode_imaging', views.mode_imaging, name='mode_imaging'),

    path('mode_test', views.mode_test, name='mode_test'),

]