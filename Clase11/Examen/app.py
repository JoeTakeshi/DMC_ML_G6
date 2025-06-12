import streamlit as st
import requests
import matplotlib.pyplot as plt

# Configuración de página
st.set_page_config(page_title="Forecast Aprobación Préstamo", layout="centered")
st.title("Aprobación de Préstamo")
st.markdown("Visualice el score y riesgo de un cliente clave para la fintech")

# Inputs del usuario
age = st.number_input("Edad", min_value=18, max_value=90, value=35)
monthly_income = st.number_input("Ingreso mensual (USD)", min_value=0.0, max_value=7000.0, value=950.0)
app_usage_score = st.slider("Score de uso del app (0–10)", min_value=0.0, max_value=10.0, value=2.0)
digital_profile_strength = st.slider("Score de perfil digital (0–100)", min_value=0.0, max_value=100.0, value=20.0)
num_contacts_uploaded = st.number_input("Número de contactos sincronizados", min_value=0, max_value=70, value=20)
residence_risk_zone = st.radio("Zona de residencia:", ["baja", "media", "alta"])
political_event_last_month = st.radio("¿Hubo disturbios/elecciones recientes en su zona?", ["Sí", "No"])

# Botón de predicción
if st.button("Predecir aprobación de crédito"):
    with st.spinner("Consultando modelo..."):
        payload = {
            "age": age,
            "monthly_income_usd": monthly_income,
            "app_usage_score": app_usage_score,
            "digital_profile_strength": digital_profile_strength,
            "num_contacts_uploaded": num_contacts_uploaded,
            "residence_risk_zone": residence_risk_zone,
            "political_event_last_month": 1 if political_event_last_month == "Sí" else 0
        }

        try:
            r = requests.post("http://localhost:8000/predict_loan", json=payload)
            if r.status_code == 200:
                resultado = r.json()
                aprobado = resultado["predicciones"]["approved"]
                score = resultado["predicciones"]["score"]

                st.success("✅ Predicción generada exitosamente")
                if aprobado:
                    st.markdown(f"### 🟢 Crédito aprobado con score: **{round(score, 2)}**")
                else:
                    st.markdown(f"### 🔴 Crédito rechazado con score: **{round(score, 2)}**")

                # Visualización de barra de score
                st.progress(min(int(score * 100), 100))
            else:
                st.error("❌ Error en la respuesta del servidor.")
        except Exception as e:
            st.error(f"❌ No se pudo conectar al API: {e}")

