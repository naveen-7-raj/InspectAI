# ==============================================================================
# InspectAI - CPU-Optimized Container Image
# ==============================================================================

FROM python:3.11-slim

# Prevent Python bytecode compilation and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install minimal OS dependencies for health probes and C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU wheel first to prevent pulling ~4GB CUDA dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application modules, configurations, and assets
COPY configs/ ./configs/
COPY src/ ./src/
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY models/ ./models/
COPY data/ ./data/

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Default startup command (FastAPI prediction backend)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

