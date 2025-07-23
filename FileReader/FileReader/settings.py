from pathlib import Path
import os

# ✅ Make BASE_DIR a Path object
BASE_DIR = Path(__file__).resolve().parent.parent

# ✅ MEDIA and STATIC paths
MEDIA_ROOT = BASE_DIR / 'media'
MEDIA_URL = '/media/'

STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static']

# ✅ SECRET_KEY and DEBUG
SECRET_KEY = 'django-insecure-vj1rud37wngvgw11%3(b12-*m3)_2a0x=!*b8x%er3dr5!s6#+'
DEBUG = True

ALLOWED_HOSTS = ['127.0.0.1', 'filereaderapp-production.up.railway.app', 'localhost']

# ✅ API keys from environment
LOCATION_KEY = os.getenv('LOCATION_KEY')
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
CSE_ID = os.getenv('CSE_ID')

# ✅ Installed apps
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    "myFileReader",
]

# ✅ Middleware
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# ✅ URL config
ROOT_URLCONF = 'FileReader.urls'

# ✅ Templates
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / "templates"],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# ✅ WSGI
WSGI_APPLICATION = 'FileReader.wsgi.application'

# ✅ Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# ✅ Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# ✅ Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# ✅ Primary key
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
