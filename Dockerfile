FROM python:3.12.10-slim

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

# Copy requirements
COPY requirements.txt /code/

# Upgrade pip
RUN pip install --upgrade pip

# Install Torch CPU only
RUN pip install torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install all requirements (except torch)
RUN grep -v -E 'torch' requirements.txt > requirements_no_torch.txt
RUN pip install --no-cache-dir -r requirements_no_torch.txt

# Force install core dependencies explicitly
RUN pip install --no-cache-dir \
    Django==5.2.4 \
    gunicorn==23.0.0 \
    pandas==2.3.1 \
    numpy==2.1.3 \
    matplotlib==3.10.3 \
    scikit-learn==1.7.1 \
    scipy==1.16.0 \
    transformers==4.53.3 \
    spacy==3.8.7 \
    nltk==3.9.1 \
    pdfplumber==0.11.7 \
    PyPDF2==3.0.1 \
    pillow==11.3.0 \
    pycountry==24.6.1 \
    googletrans==4.0.0rc1 \
    RapidFuzz==3.13.0 \
    ipython==9.4.0 \
    requests==2.32.3 \
    beautifulsoup4==4.13.3 \
    sumy==0.11.0 \
    python-docx==1.2.0 \
    deep-translator==1.11.4 \
    beautifulsoup4>=4.12 \
    langcodes==3.5.0

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy project code
COPY . /code/

# Expose port
EXPOSE 8000

# Default command
CMD ["gunicorn", "FileReader.FileReader.wsgi:application", "--bind", "0.0.0.0:8000"]