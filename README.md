# Sesión 4 · Detector de fraude en producción

API REST para servir un modelo de detección de fraude entrenado en la Sesión 1.

## Qué hace

Recibe los datos de una transacción bancaria por HTTP y devuelve la probabilidad de que sea fraude.

## Cómo arrancar

### Opción A: GitHub Codespaces (recomendado)

1. Click en `Code` → `Codespaces` → `Create codespace on main`
2. Espera a que se construya el entorno (1-2 min la primera vez)
3. En la terminal:

```bash
python train_model.py        # Genera el modelo
uvicorn app.main:app --reload
```

4. Cuando aparezca el aviso de puerto, click en `Open in Browser` y añade `/docs` al final
5. Verás Swagger UI con la API documentada

### Opción B: Local

Necesitas Python 3.11+ y pip.

```bash
pip install -r requirements.txt
python train_model.py
uvicorn app.main:app --reload
```

Abre http://localhost:8000/docs

### Opción C: Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

Abre http://localhost:8000/docs

## Probar la API

### Desde Swagger UI

Abre `/docs`, busca `POST /predict`, click en `Try it out`, edita el ejemplo y `Execute`.

### Desde curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 250.50,
    "hour": 23,
    "merchant_category": "online",
    "is_online": 1,
    "is_foreign": 1,
    "distance_from_home_km": 150.0,
    "days_since_last_tx": 0
  }'
```

Respuesta esperada:

```json
{
  "is_fraud": true,
  "fraud_probability": 0.87,
  "model_version": "1.0.0"
}
```

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/` | Saludo |
| GET | `/health` | Estado de la API y del modelo |
| POST | `/predict` | Predicción de fraude para una transacción |
| GET | `/docs` | Documentación interactiva (Swagger UI) |

## Estructura del proyecto

```
.
├── app/
│   ├── main.py          # FastAPI app y endpoints
│   ├── schemas.py       # Pydantic models (validación)
│   └── ml_model.py      # Carga y wrapper del modelo
├── model/
│   └── fraud_model.joblib   # Modelo entrenado (generado por train_model.py)
├── tests/
│   └── test_api.py      # Tests con TestClient
├── train_model.py       # Script que entrena y guarda el modelo
├── requirements.txt     # Dependencias
├── Dockerfile           # Imagen de producción
└── .devcontainer/       # Config de GitHub Codespaces
```
