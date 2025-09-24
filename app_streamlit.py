import streamlit as st
import pandas as pd
import joblib as jb

st.set_page_config(
    page_title="klasifikasi kualitas kopi",
    page_icon="☕"
)


model = jb.load("model_klasifikasi_kopi.joblib")

st.title("☕ Klasifikasi Kualitas Kopi Berdasarkan Fitur")

kadar_kafein = st.slider("Kadar Kafein", 50.0, 200.0, 110.0)
tingkat_keasaman = st.slider("Tingkat Keasaman", 0.1, 7.0, 5.0)
jenis_proses = st.pills("Jenis Proses", ["Natural", "Honey", "Washed"])

if st.button("Prediksi", type="primary"):
    data = pd.DataFrame([[kadar_kafein, tingkat_keasaman, jenis_proses]], columns=["Kadar Kafein", "Tingkat Keasaman", "Jenis Proses"])
    prediksi = model.predict(data)[0]
    presentase = max(model.predict_proba(data)[0])
    st.success(f"Prediksi: {prediksi} dengan keyakinan {presentase * 100:.2f}%")
    st.balloons()

st.divider()

st.caption("dibuat dengan  *☕* oleh **ahnaf**")
