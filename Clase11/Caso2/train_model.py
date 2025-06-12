import pandas as pd
import mlflow
#import mlflow.pycaret
from pycaret.regression import *

# Cargar datos
df = pd.read_csv("weekly_sales_forecast.csv")

# Configurar MLflow
mlflow.set_tracking_uri("http://localhost:5000")

for target, model_name in targets.items():
    print(f"\n🔁 Entrenando modelo para: {target}...")
    mlflow.set_experiment(f"forecast_{target}")
    
    df_model = df[features + [target]]
    
    s = setup(
        data=df_model,
        target=target,
        session_id=505,
        log_experiment=True,
        experiment_name=f"forecast_{target}",
        verbose=True,
        profile=False,
        use_gpu=False
    )
    
    best_model = compare_models()
    
    # Evaluación visual y registro
    evaluate_model(best_model)
    #mlflow.pycaret.log_model(best_model, model_name)
    
    # Guardar modelo localmente
    save_model(best_model, model_name)

print("✅ Modelos entrenados y registrados correctamente.")

#mlflow ui --backend-store-uri file:./mlruns --port 5000
#python train_model.py

