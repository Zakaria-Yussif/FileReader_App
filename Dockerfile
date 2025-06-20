# Use Debian-slim instead of Alpine for glibc wheels
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=FileReader.FileReader.settings
ENV DJANGO_URLS_MODULE=FileReader.FileReader.urls

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    libmagic1 \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip

# Set working directory
WORKDIR /code

# Copy requirements and install
COPY requirements.txt /code/
# Exclude any Windows-only deps if you still have them (optional):
# RUN grep -vE 'pywin32|pywinpty' requirements.txt > temp.txt && mv temp.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /code/

# Expose port and run
EXPOSE 8000
CMD ["gunicorn", "FileReader.FileReader.wsgi:application", "--bind", "0.0.0.0:8000"]
