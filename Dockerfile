FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    APP_ORIGIN=https://snapqc.icxhnk.com \
    PUBLIC_DEMO_ENABLED=true \
    IDENTITY_PLATFORM_ENABLED=false \
    IDENTITY_ISSUER=https://accounts.icxhnk.com \
    IDENTITY_CLIENT_ID=snapqc \
    IDENTITY_REDIRECT_URI=https://snapqc.icxhnk.com/api/auth/callback

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-runtime.txt ./
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY server.py whats-new.js mark.svg ./
COPY prototype ./prototype
COPY portfolio ./portfolio
COPY models/snapqc-welding-v1.onnx models/metrics-v1.json models/confusion_matrix_normalized.png models/results.png ./models/

RUN useradd --create-home --shell /bin/sh snapqc \
    && chown -R snapqc:snapqc /app

USER snapqc

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=25s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"]

CMD ["python", "-u", "server.py"]
