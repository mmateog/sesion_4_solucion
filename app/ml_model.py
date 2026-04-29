"""
Carga el modelo entrenado desde disco y expone una interfaz limpia
para hacer predicciones desde la API.

Por que separar esto del archivo main.py:
  - main.py habla HTTP, ml_model.py habla ML.
  - Si manana cambias de RandomForest a XGBoost, solo tocas aqui.
  - Tests mas faciles: puedes mockear el modelo sin tocar la API.
"""

import logging
from pathlib import Path
import joblib
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH = Path("model/fraud_model.joblib")
FRAUD_THRESHOLD = 0.5  # Umbral para considerar una prediccion como fraude


class FraudModel:
    """Wrapper alrededor del modelo entrenado."""

    def __init__(self):
        self.model = None
        self.feature_names = []
        self.version = None

    def load(self) -> None:
        """Carga el modelo desde disco. Llamar una sola vez al arrancar."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No se encuentra el modelo en {MODEL_PATH}. "
                f"Ejecuta primero: python train_model.py"
            )
        artifact = joblib.load(MODEL_PATH)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        self.version = artifact["version"]
        logger.info(f"Modelo cargado: version={self.version}, features={len(self.feature_names)}")

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, transaction: dict) -> tuple[bool, float]:
        """
        Recibe un dict con los datos de una transaccion y devuelve:
            (is_fraud, fraud_probability)
        """
        if not self.is_loaded():
            raise RuntimeError("Modelo no cargado")

        # Convertir el dict a un DataFrame de una fila
        df = pd.DataFrame([transaction])
        # Aplicar one-hot encoding igual que en el entrenamiento
        df = pd.get_dummies(df, columns=["merchant_category"], drop_first=True)
        # Asegurar que TODAS las columnas que espera el modelo estan presentes,
        # con 0 si no aparecen en este registro concreto
        df = df.reindex(columns=self.feature_names, fill_value=0)

        # predict_proba devuelve probabilidades para cada clase
        # [:, 1] = probabilidad de la clase positiva (fraude)
        prob = float(self.model.predict_proba(df)[0, 1])
        is_fraud = prob >= FRAUD_THRESHOLD

        return is_fraud, prob


# Instancia unica que importan los endpoints
fraud_model = FraudModel()
