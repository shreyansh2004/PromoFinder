# Web app image (Render, or any Docker host). Installs only the runtime libraries.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    ALLOW_LOCAL_PATH=0
WORKDIR $HOME/app

COPY --chown=user requirements-app.txt .
RUN pip install --no-cache-dir --user -r requirements-app.txt

COPY --chown=user . .
RUN mkdir -p uploads outputs feedback .cache

# Render (and most hosts) pass the port in $PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
