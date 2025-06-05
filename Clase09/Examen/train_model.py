from pycaret.classification import * 
#import mlflow
import pandas as pd
#activar el mlflow
#mlflow.set_experiment("banking-autoML")
df = pd.read_csv("C:/Users/GGjoe/OneDrive/Documentos/Visual Studio/test_prueba/DMC_ML_G6/Clase09/Examen/hbr_caso_cliente_responde_oferta.csv")
#Iniciarmos el pipeline de pycaret
s = setup(data=df,
          target='respondio_oferta',
          session_id=123,
          normalize=True,
          log_experiment=True,
          experiment_name="responde-oferta-autoML",
          categorical_features=['cliente_id','genero','nivel_educacion','region'],
          log_plots=True
          )
#Comparar modelos y el registro en el mlflow
best_model = compare_models()
final_model = tune_model(best_model)
save_model(final_model,"modelo_responde_oferta")