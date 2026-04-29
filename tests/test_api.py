"""
Tests basicos de la API. Ejecutar con:
    pytest

TestClient simula peticiones HTTP sin levantar el servidor de verdad,
asi los tests son rapidos y no necesitan que la API este arrancada.

OJO: usamos TestClient como context manager (`with`) para que el
lifespan de FastAPI se ejecute (que carga el modelo).
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Cliente de prueba con lifespan activado."""
    with TestClient(app) as c:
        yield c


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "fraud-detection-api"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] is not None


def test_predict_legitimate(client):
    """Una transaccion mundana deberia clasificarse como NO fraude."""
    response = client.post("/predict", json={
        "amount": 12.50,
        "hour": 14,
        "merchant_category": "grocery",
        "is_online": 0,
        "is_foreign": 0,
        "distance_from_home_km": 1.5,
        "days_since_last_tx": 1,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["is_fraud"] is False
    assert body["fraud_probability"] < 0.5


def test_predict_suspicious(client):
    """Una transaccion con todas las banderas rojas deberia ser fraude."""
    response = client.post("/predict", json={
        "amount": 3500.00,
        "hour": 3,
        "merchant_category": "electronics",
        "is_online": 1,
        "is_foreign": 1,
        "distance_from_home_km": 800.0,
        "days_since_last_tx": 0,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["is_fraud"] is True
    assert body["fraud_probability"] > 0.5


def test_predict_invalid_hour(client):
    """Hora fuera de rango deberia devolver 422."""
    response = client.post("/predict", json={
        "amount": 100,
        "hour": 25,  # invalido
        "merchant_category": "grocery",
        "is_online": 0,
        "is_foreign": 0,
        "distance_from_home_km": 1.0,
        "days_since_last_tx": 1,
    })
    assert response.status_code == 422


def test_predict_invalid_category(client):
    """Categoria que el modelo no conoce."""
    response = client.post("/predict", json={
        "amount": 100,
        "hour": 12,
        "merchant_category": "unicorn",  # no existe
        "is_online": 0,
        "is_foreign": 0,
        "distance_from_home_km": 1.0,
        "days_since_last_tx": 1,
    })
    assert response.status_code == 422


def test_predict_missing_field(client):
    """Falta un campo obligatorio."""
    response = client.post("/predict", json={
        "amount": 100,
        # falta hour
        "merchant_category": "grocery",
        "is_online": 0,
        "is_foreign": 0,
        "distance_from_home_km": 1.0,
        "days_since_last_tx": 1,
    })
    assert response.status_code == 422
