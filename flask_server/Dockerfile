FROM python:3.10-slim

WORKDIR /app

# System deps for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir flask

# Install sentencepiece for T5 models
RUN pip install --no-cache-dir sentencepiece

# Pre-download the model (choose one):

# Option 1: Flan-T5-small (fastest, ~80MB)
RUN python -c "from transformers import T5ForConditionalGeneration, T5Tokenizer; \
    print('Downloading Flan-T5-small model...'); \
    T5Tokenizer.from_pretrained('google/flan-t5-small'); \
    T5ForConditionalGeneration.from_pretrained('google/flan-t5-small'); \
    print('Model downloaded successfully')" \

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
