# Versao web do Libre-se (servidor + pagina), para o Railway ou qualquer Docker.
# Antes de publicar: python servidor/fixar_versoes.py  (no .venv em que os modelos foram treinados)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    LIBRES_DADOS=/data

WORKDIR /app
COPY servidor/requirements.txt servidor/requirements-modelos.txt servidor/
RUN pip install -r servidor/requirements.txt -r servidor/requirements-modelos.txt

COPY . .
# MediaPipe para navegador (npm) e o modelo do corpo, se ainda nao estiverem na pasta
RUN python servidor/preparar_web.py

# a calibracao de cada pessoa fica em /data (no Railway: monte um volume em /data)
CMD ["sh", "-c", "exec uvicorn servidor.app:app --host 0.0.0.0 --port ${PORT:-8000} --ws-max-size 4000000 --proxy-headers --forwarded-allow-ips='*'"]
