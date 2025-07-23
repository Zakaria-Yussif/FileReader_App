FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    libmagic1 \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /code

# Copy requirements file
COPY requirements.txt /code/

# Upgrade pip and install torch separately (CPU-only)
RUN pip install --upgrade pip
RUN pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Exclude torch and incompatible ipython version from requirements
RUN grep -v -E 'torch|ipython==9.0.1' requirements.txt > requirements_no_torch.txt

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements_no_torch.txt

# Copy application source code
COPY . /code/

# Expose port
EXPOSE 8000

# Start Gunicorn server
CMD ["gunicorn", "FileReader.FileReader.wsgi:application", "--bind", "0.0.0.0:8000"]
