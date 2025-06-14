FROM python:3.11-slim

# Install system dependencies if needed (e.g., libmagic, libpoppler)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    libmagic-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt requirements-core.txt requirements-data.txt requirements-files.txt requirements-utils.txt /code/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /code/

CMD ["gunicorn", "FileReader.wsgi:application", "--bind", "0.0.0.0:8000"]
