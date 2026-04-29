"""
API REST para servir el modelo de deteccion de fraude.

Para arrancar localmente:
    uvicorn app.main:app --reload

Documentacion interactiva en: http://localhost:8000/docs
"""

import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from app.ml_model import fraud_model
from app.schemas import TransactionInput, PredictionOutput, HealthResponse

# Logging basico - en produccion querras estructurarlo mejor (JSON, niveles, etc)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fraud-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Se ejecuta una vez al arrancar la API y otra al apagarla.
    Cargamos el modelo al arrancar para que la primera peticion no tarde mas.
    """
    logger.info("Arrancando API de deteccion de fraude...")
    fraud_model.load()
    logger.info("Modelo cargado y listo")
    yield
    logger.info("Apagando API...")


app = FastAPI(
    title="Fraud Detection API",
    description="API para detectar transacciones fraudulentas con un modelo de ML",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    """Saludo basico."""
    return {
        "service": "fraud-detection-api",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Endpoint para que un orquestador (Kubernetes, AWS ECS, etc.)
    sepa si la API esta sana. Si esto no devuelve 200, el orquestador
    asume que el contenedor esta roto y lo reinicia.
    """
    if fraud_model.is_loaded():
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=fraud_model.version,
        )
    return HealthResponse(status="down", model_loaded=False)


@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):
    """
    Recibe los datos de una transaccion y predice si es fraude.

    El modelo devuelve una probabilidad. Si supera el umbral (0.5 por defecto),
    se considera fraude.
    """
    start = time.perf_counter()

    try:
        is_fraud, prob = fraud_model.predict(transaction.model_dump())
    except Exception as e:
        logger.exception("Error en prediccion")
        raise HTTPException(status_code=500, detail=f"Error en prediccion: {e}")

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"prediction: amount={transaction.amount} category={transaction.merchant_category} "
        f"-> is_fraud={is_fraud} prob={prob:.3f} ({elapsed_ms:.1f}ms)"
    )

    return PredictionOutput(
        is_fraud=is_fraud,
        fraud_probability=round(prob, 4),
        model_version=fraud_model.version,
    )
