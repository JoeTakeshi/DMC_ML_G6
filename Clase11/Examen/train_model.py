import pandas as pd
from pycaret.classification import *
import mlflow
#import mlflow.pycaret

# Cargar dataset
df = pd.read_csv("fintech_credit_approval.csv")

# Excluir columnas no predictivas
df_model = df.drop(columns=["user_id"])

# Configurar MLflow local
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fintech_credit")

# Configurar PyCaret
s = setup(
    data=df_model,
    target="approved",
    session_id=123,
    log_experiment=True,
    experiment_name="fintech_credit",
    categorical_features=['residence_risk_zone'],
    verbose=True,
    profile=False,
    use_gpu=False,
    fix_imbalance=True, 
    fix_imbalance_method='smote'
)

# Entrenar modelos y seleccionar el mejor
best_model = compare_models()

# Log en MLflow
#mlflow.pycaret.log_model(best_model, "mejor_modelo_propension")

# Guardar modelo
save_model(best_model, "propension_model")

print("✅ Entrenamiento y registro completado.")


#mlflow ui --backend-store-uri file:./mlruns --port 5000
#python train_model.py
