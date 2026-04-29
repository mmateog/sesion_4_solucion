"""
Entrena un Random Forest para detectar fraude y lo guarda en disco.
Es el mismo modelo que se construyó en la Sesion 1, ahora persistido
para que la API lo pueda cargar.

Ejecutar una sola vez:
    python train_model.py

Genera:
    model/fraud_model.joblib
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_PATH = Path("model/fraud_model.joblib")
MODEL_VERSION = "1.0.0"


def generar_dataset(n=10_000, seed=42):
    """Genera un dataset sintetico de transacciones bancarias (~5% fraude)."""
    rng = np.random.default_rng(seed)
    n_fraud = int(n * 0.05)
    n_legit = n - n_fraud

    # Transacciones legitimas
    legit = pd.DataFrame({
        "amount": rng.lognormal(mean=3.5, sigma=1.0, size=n_legit),
        "hour": rng.choice(range(9, 23), size=n_legit),
        "merchant_category": rng.choice(
            ["grocery", "restaurant", "gas", "online", "electronics", "travel", "atm"],
            size=n_legit, p=[0.3, 0.2, 0.15, 0.15, 0.08, 0.07, 0.05]
        ),
        "is_online": rng.choice([0, 1], size=n_legit, p=[0.6, 0.4]),
        "is_foreign": rng.choice([0, 1], size=n_legit, p=[0.92, 0.08]),
        "distance_from_home_km": rng.lognormal(mean=1.2, sigma=1.0, size=n_legit),
        "days_since_last_tx": rng.exponential(scale=2.0, size=n_legit),
        "is_fraud": 0,
    })

    # Transacciones fraudulentas
    # Pesos por hora: madrugada con peso alto, dia con peso bajo
    # Suma debe ser 1.0
    hour_weights = [0.10] * 6 + [0.05] * 2 + [0.01875] * 16  # 0.6 + 0.1 + 0.3 = 1.0
    fraud = pd.DataFrame({
        "amount": rng.lognormal(mean=5.5, sigma=1.2, size=n_fraud),
        "hour": rng.choice(range(0, 24), size=n_fraud, p=hour_weights),
        "merchant_category": rng.choice(
            ["grocery", "restaurant", "gas", "online", "electronics", "travel", "atm"],
            size=n_fraud, p=[0.05, 0.05, 0.10, 0.40, 0.20, 0.15, 0.05]
        ),
        "is_online": rng.choice([0, 1], size=n_fraud, p=[0.25, 0.75]),
        "is_foreign": rng.choice([0, 1], size=n_fraud, p=[0.65, 0.35]),
        "distance_from_home_km": rng.lognormal(mean=4.0, sigma=1.0, size=n_fraud),
        "days_since_last_tx": rng.exponential(scale=2.0, size=n_fraud),
        "is_fraud": 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def main():
    print("Generando dataset...")
    df = generar_dataset()
    print(f"  Total: {len(df)} transacciones")
    print(f"  Fraude: {int(df['is_fraud'].sum())} ({df['is_fraud'].mean():.1%})")

    print("\nPreparando features...")
    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud"])
    X = pd.get_dummies(X, columns=["merchant_category"], drop_first=True)
    feature_names = X.columns.tolist()
    print(f"  Features: {len(feature_names)}")

    print("\nEntrenando modelo...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("\nEvaluando...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Legitima", "Fraude"]))

    # Guardar modelo + metadata juntos
    artifact = {
        "model": model,
        "feature_names": feature_names,
        "version": MODEL_VERSION,
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModelo guardado en: {MODEL_PATH}")
    print(f"Tamano: {MODEL_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
