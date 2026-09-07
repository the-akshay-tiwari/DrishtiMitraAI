# DrishtiMitra Production Multi-Stage Deployment Dockerfile

# Stage 1: Build Frontend static bundle
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . ./
RUN npm run build

# Stage 2: Python FastAPI inference server
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend code, compiled frontend bundle, and baseline model checkpoint
COPY backend/ /app/backend/
COPY --from=frontend-builder /app/dist /app/dist
COPY artifacts/drishtimitra_efficientnet_b0.pt /app/artifacts/drishtimitra_efficientnet_b0.pt

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
