# Dockerfile minimalista pero "production-ready"
# Imagen oficial de Python en variante slim (mas ligera que la full)
FROM python:3.11-slim

# Variables de entorno utiles
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1. Copiar SOLO requirements primero
#    Asi Docker cachea esta capa y no reinstala todas las deps
#    cuando solo cambia codigo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copiar el resto del codigo y el modelo
COPY app/ ./app/
COPY model/ ./model/

# 3. Usuario no-root por seguridad
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# 4. Documentar el puerto que la app expone (no abre puerto, es informativo)
EXPOSE 8000

# 5. Healthcheck para que Docker / orquestadores sepan si la API esta viva
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

# 6. Comando final
#    --host 0.0.0.0 es OBLIGATORIO en contenedor: si no, no acepta conexiones externas
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
