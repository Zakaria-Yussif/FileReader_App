FROM python:3.11-alpine

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=FileReader.FileReader.settings

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
    && pip install --upgrade pip

# Set work directory
WORKDIR /code

# Copy and install requirements
COPY requirements*.txt /code/
COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . /code/

# Run the server
CMD ["gunicorn", "FileReader.FileReader.wsgi:application", "--bind", "0.0.0.0:8000"]
