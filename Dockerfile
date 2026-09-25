# Minimal image for backend/ui services (CPU default).
# GPU: build FROM a CUDA runtime and install the matching torch wheel first.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY constraints.txt pyproject.toml requirements.txt ./
COPY src/ ./src/
# Install the project with the runtime extras used by backend + Streamlit UI.
# For CUDA machines, pre-install torch/torchvision from the matching cuXX index first.
RUN pip install --no-cache-dir -U 'setuptools>=69,<80' wheel \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -c constraints.txt -e '.[ui,face,store,dev]'

COPY apps/ ./apps/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY tools/ ./tools/

EXPOSE 8767 8765 8766 8501
