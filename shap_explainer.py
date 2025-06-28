import shap
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np 

def explain_with_shap(explainer, model, X_input):
    try:
        transformed = model.named_steps['prep'].transform(X_input)
        shap_values = explainer(transformed)

        st.subheader("📊 Biểu đồ SHAP (đóng góp của từng đặc trưng)")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], ax=ax, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Không thể hiển thị SHAP: {e}")

@st.cache_resource
def get_shap_explainer(_model, background_df):
    transformed_bg = _model.named_steps['prep'].transform(background_df)
    explainer = shap.Explainer(_model.named_steps['clf'], transformed_bg)  # 'clf' là bước trong pipeline
    return explainer

