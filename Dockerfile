FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for XGBoost / LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and app code
COPY setup.py .
COPY src/ src/
RUN pip install --no-cache-dir -e .

COPY app/ app/

# Copy data and models (may be empty; app falls back to demo data)
COPY data/ data/
COPY models/ models/

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
