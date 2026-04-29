"""
Pydantic schemas: definen la forma EXACTA de los datos
que la API espera recibir y devolver.

Beneficios:
  - Validacion automatica (rangos, tipos, valores permitidos)
  - Errores HTTP 422 claros si llega algo mal formado
  - Documentacion automatica en Swagger UI
"""

from pydantic import BaseModel, Field
from typing import Literal


# Categorias permitidas (las mismas que vio el modelo durante entrenamiento)
MerchantCategory = Literal[
    "grocery", "restaurant", "gas", "online", "electronics", "travel", "atm"
]


class TransactionInput(BaseModel):
    """Datos de una transaccion para predecir si es fraude."""

    amount: float = Field(..., ge=0, le=100_000, description="Importe en euros")
    hour: int = Field(..., ge=0, le=23, description="Hora del dia (0-23)")
    merchant_category: MerchantCategory = Field(..., description="Tipo de comercio")
    is_online: int = Field(..., ge=0, le=1, description="1 si la transaccion fue online")
    is_foreign: int = Field(..., ge=0, le=1, description="1 si fue en pais extranjero")
    distance_from_home_km: float = Field(..., ge=0, description="Distancia desde el domicilio del titular")
    days_since_last_tx: float = Field(..., ge=0, description="Dias desde la ultima transaccion del titular")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "amount": 250.50,
                "hour": 23,
                "merchant_category": "online",
                "is_online": 1,
                "is_foreign": 1,
                "distance_from_home_km": 150.0,
                "days_since_last_tx": 0,
            }]
        }
    }


class PredictionOutput(BaseModel):
    """Resultado de una prediccion de fraude."""

    is_fraud: bool = Field(..., description="True si el modelo cree que es fraude")
    fraud_probability: float = Field(..., ge=0, le=1, description="Probabilidad estimada de fraude")
    model_version: str = Field(..., description="Version del modelo que hizo la prediccion")

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    """Estado del servicio."""

    status: Literal["ok", "degraded", "down"]
    model_loaded: bool
    model_version: str | None = None

    model_config = {"protected_namespaces": ()}
