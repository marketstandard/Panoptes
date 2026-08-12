FROM node:22-bookworm-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PANOPTES_PROFILE=fixture
WORKDIR /app
COPY backend/pyproject.toml ./backend/pyproject.toml
COPY backend/panoptes ./backend/panoptes
COPY schemas ./schemas
COPY fixtures ./fixtures
RUN python -m pip install --no-cache-dir ./backend
COPY --from=frontend /app/frontend/dist ./frontend/dist
EXPOSE 8000
CMD ["sh", "-c", "uvicorn panoptes.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
