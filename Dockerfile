FROM python:3.11-alpine

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=FileReader.FileReader.settings
ENV DJANGO_URLS_MODULE=FileReader.FileReader.urls

# Install dependencies
RUN apk update && apk add --no-cache \
    build-base \
    poppler-utils \
    libmagic \
    libffi-dev \
    musl-dev \
    gcc \
    python3-dev \
    jpeg-dev \
    zlib-dev \
    tzdata \
    && pip install --upgrade pip

# Set working directory
WORKDIR /code

# Copy and install requirements
COPY requirements.txt /code/

# Ensure incompatible Windows packages are skipped
# Remove any Windows-only packages like pywin32 and pywinpty from requirements.txt first!
RUN grep -vE 'pywin32|pywinpty' requirements.txt > temp.txt && mv temp.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . /code/

# Run the server
CMD ["gunicorn", "FileReader.FileReader.wsgi:application", "--bind", "0.0.0.0:8000"]
