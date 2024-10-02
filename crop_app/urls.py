from django.urls import path
from .views import generate_report, generate_pdf, home, get_wordpress_posts, capture_image, predict_disease, MyDataView

from crop_app import views

urlpatterns = [
    path('generate_pdf/', generate_pdf, name='generate_pdf'),
    path('generate-report/', generate_report, name='generate_report'),
    path('home/', home, name='home'),
    path('wordpress-posts/', get_wordpress_posts, name='wordpress-posts'),
    path('capture_image/', capture_image, name='capture_image'),
    path('', predict_disease, name='predict_disease'),
    path('my-data/', MyDataView.as_view(), name='my-data'),
]
