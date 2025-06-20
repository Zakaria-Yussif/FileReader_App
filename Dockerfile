FROM python:3.10-slim

# system deps ...
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential poppler-utils libmagic1 libffi-dev libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt /code/

# Install torch separately with the official index URL
RUN pip install --upgrade pip
RUN pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Then install rest of the requirements excluding torch
RUN grep -v 'torch' requirements.txt > requirements_no_torch.txt
RUN pip install --no-cache-dir -r requirements_no_torch.txt

COPY . /code/

EXPOSE 8000
CMD ["gunicorn", "FileReader.FileReader.wsgi:application", "--bind", "0.0.0.0:8000"]
