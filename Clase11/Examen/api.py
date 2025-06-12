from fastapi import FastAPI
from pydantic import BaseModel
from pycaret.classification import load_model, predict_model
import pandas as pd

# Inicializar API
app = FastAPI(title="API de Forecast Aprobación Prestamo")

# Cargar modelos
model_loan = load_model("propension_model")

# Clase de entrada
class cliente(BaseModel):
    age: int
    monthly_income_usd: float
    app_usage_score: float
    digital_profile_strength: float
    num_contacts_uploaded: int
    residence_risk_zone: str
    political_event_last_month: int

# Endpoint para predecir 
@app.post("/predict_loan")
def predict(cliente: cliente):
    data = pd.DataFrame([cliente.dict()])
    result = predict_model(model_loan, data=data)
    approved = int(result["prediction_label"][0])
    score = float(result["prediction_score"][0])

    return {
        "input": cliente.dict(),
        "predicciones": {
            "approved": approved,
            "score": round(score, 4)
        }
    }

#pip install fastapi uvicorn
#uvicorn api:app --reload --port 8000
