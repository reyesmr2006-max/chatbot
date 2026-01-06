import streamlit as st
import pickle
import numpy as np

# Cargar el modelo
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Detector de Sentimientos 🧠💬")
st.write("Analiza si un texto es positivo, negativo o neutral")

texto = st.text_area("Escribe un texto:")

if st.button("Analizar"):
    if texto.strip() == "":
        st.warning("Por favor escribe un texto")
    else:
        proba = model.predict_proba([texto])[0]
        clases = model.classes_
        idx = np.argmax(proba)

        st.success(f"Sentimiento: {clases[idx]}")
        st.write(f"Confianza: {proba[idx]*100:.2f}%")


              
