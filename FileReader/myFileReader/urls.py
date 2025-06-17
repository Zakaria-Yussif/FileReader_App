from django.contrib import admin
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", views.index, name="index"),
    # path("login/", views.user_login, name="login"),
    # path("register/", views.register, name="register"),
    path("upload_datafiles/", views.upload_datafiles, name="upload_datafiles"),  # Separate upload route
    path("submit_message/", views.submit_message, name="submit_message"),
    path('clear_message_data/',views.clear_message_data, name='clear_message_data'),
    # Fix path
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)