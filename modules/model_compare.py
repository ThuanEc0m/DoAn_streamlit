import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy

from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix,
    roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import shap

# Danh sách mô hình mặc định
MODEL_DICT = {
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Tối ưu bằng GridSearchCV
def optimize_hyperparameters(model_name, X, y):
    if model_name == "Random Forest":
        model = RandomForestClassifier()
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [None, 5, 10]
        }
    elif model_name == "XGBoost":
        model = XGBClassifier(eval_metric="logloss", use_label_encoder=False)
        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [3, 6],
            "learning_rate": [0.01, 0.1]
        }
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        param_grid = {
            "C": [0.1, 1.0, 10.0],
            "solver": ["liblinear", "lbfgs"]
        }
    elif model_name == "CatBoost":
        model = CatBoostClassifier(verbose=0)
        param_grid = {
            "depth": [4, 6],
            "learning_rate": [0.01, 0.1]
        }
    elif model_name == "LightGBM":
        model = LGBMClassifier()
        param_grid = {
            "n_estimators": [50, 100],
            "num_leaves": [31, 63]
        }
    else:
        return None

    grid = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_

# Huấn luyện + SHAP + ROC
def train_and_evaluate(models, X_train, y_train, X_test, y_test, show_shap):
    y_train = pd.Series(y_train).astype(int).values
    y_test = pd.Series(y_test).astype(int).values

    results = []
    roc_fig, ax = plt.subplots(figsize=(5, 4))
    shap_charts = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        results.append({
            "Model": name,
            "F1-Score": f1,
            "Precision": precision,
            "Recall": recall,
            "AUC": roc_auc
        })

        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

        if show_shap:
            try:
                if isinstance(model, (RandomForestClassifier, XGBClassifier, LGBMClassifier, CatBoostClassifier)):
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test[:100])
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                fig, _ = plt.subplots(figsize=(4, 3))
                shap.summary_plot(shap_values.values, X_test[:100], show=False)
                shap_charts.append((name, fig))
            except Exception as e:
                st.error(f"❌ SHAP lỗi với mô hình {name}: {e}")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    return pd.DataFrame(results), roc_fig, shap_charts

# Vẽ confusion matrix lưới
def plot_confusion_matrix_grid(models, X_test, y_test):
    cols = st.columns(len(models))
    for i, (name, model) in enumerate(models.items()):
        model = copy.deepcopy(model)
        model.fit(X_test, y_test)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(3, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        cols[i].pyplot(fig)

# Trang chính
def render_comparison_page():
    st.header("📈 So sánh các mô hình ML")

    X_train = st.session_state.get("X_train")
    X_test = st.session_state.get("X_test")
    y_train = st.session_state.get("y_train")
    y_test = st.session_state.get("y_test")

    if X_train is None or y_train is None:
        st.warning("⚠️ Vui lòng nạp dữ liệu trước khi so sánh mô hình.")
        return

    selected_models = st.multiselect(
        "Chọn các mô hình để so sánh:",
        list(MODEL_DICT.keys()),
        default=["Random Forest", "XGBoost"]
    )

    use_grid = st.checkbox("⚙️ Tối ưu hyperparameter (GridSearchCV)")
    show_conf_matrix = st.checkbox("📉 Hiển thị Confusion Matrix")
    show_shap = st.checkbox("📌 Dùng SHAP để giải thích")

    if selected_models:
        models = {}
        for name in selected_models:
            if use_grid:
                with st.spinner(f"🔍 Đang tối ưu {name}..."):
                    models[name] = optimize_hyperparameters(name, X_train, y_train)
                    st.success(f"✅ {name} đã tối ưu")
            else:
                if name == "Random Forest":
                    models[name] = RandomForestClassifier()
                elif name == "XGBoost":
                    models[name] = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
                elif name == "LightGBM":
                    models[name] = LGBMClassifier()
                elif name == "CatBoost":
                    models[name] = CatBoostClassifier(verbose=0)
                elif name == "Logistic Regression":
                    models[name] = LogisticRegression(max_iter=1000)

        results_df, roc_fig, shap_charts = train_and_evaluate(
            models, X_train, y_train, X_test, y_test, show_shap
        )

        st.subheader("🔍 Bảng đánh giá")
        float_cols = results_df.select_dtypes(include="number").columns
        st.dataframe(results_df.style.format({col: "{:.3f}" for col in float_cols}))

        st.subheader("📊 ROC Curve")
        st.pyplot(roc_fig)

        if show_conf_matrix:
            st.subheader("📉 Confusion Matrix")
            plot_confusion_matrix_grid(models, X_test, y_test)

        if show_shap and shap_charts:
            st.subheader("📌 Biểu đồ SHAP")
            cols = st.columns(len(shap_charts))
            for i, (name, fig) in enumerate(shap_charts):
                with cols[i]:
                    st.markdown(f"**{name}**")
                    st.pyplot(fig)
