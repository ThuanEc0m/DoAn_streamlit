import streamlit as st
import pandas as pd
import cloudpickle
import os
import sqlalchemy
import requests
from streamlit_echarts import st_echarts
import matplotlib.pyplot as plt
import seaborn as sns
import ui_theme
import altair as alt
import datetime
import monitor
import streamlit.components.v1 as components
from ydata_profiling import ProfileReport
from ui_theme import render_sidebar_lottie
from shap_explainer import get_shap_explainer, explain_with_shap



# GỌI NGAY ĐẦU TIÊN
st.set_page_config(
    page_title="Phân tích & Dự đoán Marketing",
    layout="wide",
    page_icon=":bar_chart:"
)

#model_compare------------------------------
from modules import model_compare

#--gọi gif__json
render_sidebar_lottie("assets/pew.json")
# Load model
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", 'rb') as f:
            return cloudpickle.load(f)
    except Exception as e:
        st.error(f"Không tải được model: {e}")
        return None
model = load_model()
feature_names = model.named_steps['prep'].feature_names_in_ if model else []

FIELD_DISPLAY = {
    'age': 'Tuổi', 'job': 'Nghề nghiệp', 'marital': 'Hôn nhân', 'education': 'Học vấn',
    'default': 'Nợ xấu', 'balance': 'Số dư TK', 'housing': 'Vay nhà', 'loan': 'Vay tiêu dùng',
    'contact': 'Liên hệ', 'day': 'Ngày gọi', 'month': 'Tháng gọi', 'duration': 'Thời lượng gọi (s)',
    'campaign': 'Số lần gọi (chiến dịch)', 'pdays': 'Số ngày từ lần gọi trước',
    'previous': 'Số lần gọi trước', 'poutcome': 'KQ chiến dịch trước', 'deposit': 'Mở sổ tiết kiệm'
}
# Load đa nguồn
@st.cache_data(show_spinner=False)
def load_file(file):
    if file.name.endswith(".csv"): return pd.read_csv(file)
    return pd.read_excel(file, engine="openpyxl")

@st.cache_data(show_spinner=False)
def load_mysql(host, port, user, pwd, db, table):
    uri = f"mysql+mysqlconnector://{user}:{pwd}@{host}:{port}/{db}"
    eng = sqlalchemy.create_engine(uri)
    return pd.read_sql_table(table, eng)

@st.cache_data(show_spinner=False)
def load_postgres(host, port, user, pwd, db, table):
    uri = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
    eng = sqlalchemy.create_engine(uri)
    return pd.read_sql_table(table, eng)

@st.cache_data(show_spinner=False)
def load_api(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return pd.json_normalize(data)

# Auto EDA: YData Profiling
def run_ydata_profiling(df):
    st.subheader("📊 Auto-EDA với YData Profiling")
    profile = ProfileReport(df, title="Report", explorative=True)
    profile_html = profile.to_html()
    components.html(profile_html, height=1000, scrolling=True)
def run_sweetviz(df):
    import sweetviz as sv
    report = sv.analyze(df)
    report.show_html("sweetviz_report.html", open_browser=False)
    with open("sweetviz_report.html", 'r', encoding='utf-8') as f:
        html = f.read()
    components.html(html, height=1000, scrolling=True)

# Main App
def main():
    ui_theme.set_custom_theme()
    ui_theme.render_time_top_right()
    ui_theme.render_title_with_image("assets/data1.png", "Hệ thống Phân tích & Dự đoán mở sổ tiết kiệm")


    
# Sau đó mới đến phần chọn nguồn dữ liệu
    st.sidebar.header(" 1️⃣ Chọn nguồn dữ liệu")
    source = st.sidebar.selectbox("Data source", ["File Upload", "MySQL", "PostgreSQL", "API"])
    df = None
    if source == "File Upload":
        f = st.sidebar.file_uploader("Chọn file CSV/Excel", type=["csv", "xls", "xlsx"])
        if f: df = load_file(f)
    elif source == "MySQL":
        host = st.sidebar.text_input("MySQL host", "localhost")
        port = st.sidebar.text_input("Port", "3306")
        user = st.sidebar.text_input("User", "")
        pwd  = st.sidebar.text_input("Password", "", type="password")
        db   = st.sidebar.text_input("Database", "")
        table= st.sidebar.text_input("Table", "")
        if st.sidebar.button("Load MySQL"):
            try:
                df = load_mysql(host, port, user, pwd, db, table)
                st.sidebar.success("✅ Load thành công")
            except Exception as e:
                st.sidebar.error(f"Lỗi: {e}")
    elif source == "PostgreSQL":
        host = st.sidebar.text_input("Postgres host", "localhost")
        port = st.sidebar.text_input("Port", "5432")
        user = st.sidebar.text_input("User", "")
        pwd  = st.sidebar.text_input("Password", "", type="password")
        db   = st.sidebar.text_input("Database", "")
        table= st.sidebar.text_input("Table", "")
        if st.sidebar.button("Load Postgres"):
            try:
                df = load_postgres(host, port, user, pwd, db, table)
                st.sidebar.success("✅ Load thành công")
            except Exception as e:
                st.sidebar.error(f"Lỗi: {e}")
    else:
        url = st.sidebar.text_input("API URL endpoint", "")
        if st.sidebar.button("Load API"):
            try:
                df = load_api(url)
                st.sidebar.success("✅ API loaded!")
            except Exception as e:
                st.sidebar.error(f"Lỗi: {e}")

        # Nếu dữ liệu đã nạp và có cột 'deposit' → chia train-test
    if df is not None and 'deposit' in df.columns:
        from sklearn.model_selection import train_test_split

        df_model = df.dropna(subset=['deposit'])  # loại bỏ dòng thiếu nhãn nếu có
        X = df_model.drop(columns=['deposit'])
        # Encode các cột object/category
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = X[col].astype('category').cat.codes
        y = df_model['deposit'].replace({'yes': 1, 'no': 0})  # chuyển nhãn thành nhị phân

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
        except Exception as e:
            st.warning(f"⚠️ Không thể tách train/test: {e}")
    if df is not None and not df.empty:
        st.sidebar.success(f"✅ Dữ liệu đã nạp: {df.shape[0]} dòng, {df.shape[1]} cột")

    #    --- Khởi tạo session nếu chưa có ---
    if "current_page" not in st.session_state:
        st.session_state.current_page = "🏠 Trang chủ"
    # --- Callback khi chọn mục Thông tin ---
    def on_info_change():
        st.session_state.current_page = st.session_state.info_select
    # --- Callback khi chọn mục Chức năng ---
    def on_func_change():
        st.session_state.current_page = st.session_state.func_select
    # --- 2️⃣ Thông tin ---
    st.sidebar.header("2️⃣ Thông tin")
    st.sidebar.selectbox(
        "📘 Chọn mục",
        [
            "🏠 Trang chủ",
            "📌 Câu hỏi nghiên cứu",
            "📚 Nguồn dữ liệu & mô tả",
            "📊 Phân tích & kết luận"
        ],
        key="info_select",
        on_change=on_info_change
    )
    # --- 3️⃣ Chức năng ---
    st.sidebar.header("3️⃣ Chức năng")
    st.sidebar.selectbox(
        "⚙️ Chọn chức năng",
        [
            "📊 Báo cáo",
            "🤖 Dự đoán",
            "🔬 So sánh",
            "📈 Theo dõi mô hình",
            "🧩 Phân tích EDA tự động",
            "📈 Model comparison"
        ],
        key="func_select",
        on_change=on_func_change
    )
# --- Trang hiện tại là cái vừa được chọn ---
    page = st.session_state.current_page

    # Nếu chưa có dữ liệu và không ở các mục markdown thì cảnh báo
    if df is None and page not in [
        "🏠 Trang chủ", "📌 Câu hỏi nghiên cứu", "📚 Nguồn dữ liệu & mô tả", "📊 Phân tích & kết luận"]:
        st.info("Chưa có dữ liệu, vui lòng nạp file.")
        return

    # 1. Các mục Markdown
    elif page in [
        "🏠 Trang chủ", "📌 Câu hỏi nghiên cứu", "📚 Nguồn dữ liệu & mô tả", "📊 Phân tích & kết luận"]:
        file_map = {
            "🏠 Trang chủ": "vanban/trangchu.md",
            "📌 Câu hỏi nghiên cứu": "vanban/cauhoinghiencuu.md",
            "📚 Nguồn dữ liệu & mô tả": "vanban/nguondulieuvamota.md",
            "📊 Phân tích & kết luận": "vanban/phantichvaketluan.md",
        }
        with open(file_map[page], "r", encoding="utf-8") as f:
            lines = f.readlines()

        section_title = None
        section_body = []
        sections = []

        for line in lines:
            if line.startswith("## "):
                if section_title:
                    sections.append((section_title, "".join(section_body)))
                section_title = line.strip("## ").strip()
                section_body = []
            else:
                section_body.append(line)
        if section_title:
            sections.append((section_title, "".join(section_body)))

        st.title(page)
        for title, content in sections:
            with st.expander(f" {title}"):
                st.markdown(content, unsafe_allow_html=True)
# 2. Mục báo cáo tổng quan
    elif page == "📈 Báo cáo tổng quan":
        st.markdown("### Tổng quan dữ liệu")
        total = len(df)
        if 'deposit' in df.columns:
            opened = df['deposit'].isin(['yes', 1]).sum()
            not_opened = total - opened
            ratio = (opened / total) * 100 if total > 0 else 0
            st.info(f"""
                - **Tổng khách:** {total}
                - **Có mở sổ:** {opened}
                - **Chưa mở:** {not_opened}
                - **Tỉ lệ mở:** {ratio:.2f}%
            """)
        else:
            st.warning("⛔ File không có cột deposit – không thể phân tích tỷ lệ mở sổ.")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("#### 📌 Biểu đồ đếm")
            cat_col = st.selectbox("Chọn cột phân loại", cat_cols)
            cat_count = df[cat_col].value_counts().reset_index()
            cat_count.columns = [cat_col, 'Số lượng']
            bar_chart = alt.Chart(cat_count).mark_bar().encode(
                x=alt.X(f'{cat_col}:N', sort='-y', title='Giá trị'),
                y=alt.Y('Số lượng:Q', title='Số lượng'),
                tooltip=[cat_col, 'Số lượng']
            ).properties(width=380, height=300)
            st.altair_chart(bar_chart)
        with col2:
            st.markdown("#### 📌 Histogram (cột số)")
            num_col = st.selectbox("Chọn cột số", num_cols)
            if num_col:
                hist_data = df[[num_col]].dropna()
                hist_chart = alt.Chart(hist_data).mark_bar().encode(
                    alt.X(f"{num_col}:Q", bin=alt.Bin(maxbins=30), title="Giá trị"),
                    y=alt.Y('count()', title="Số lượng"),
                    tooltip=[alt.Tooltip('count()', title="Số lượng")]
                ).properties(width=380, height=300)
                st.altair_chart(hist_chart)
        if 'deposit' in df.columns:
            col3, col4 = st.columns(2, gap="large")
            with col3:
                st.markdown("#### 🧑‍💼 Tỉ lệ mở sổ theo nghề")
                if 'job' in df.columns:
                    job_df = df.groupby('job')['deposit'].value_counts(normalize=True).unstack().fillna(0)*100
                    chart = alt.Chart(job_df.reset_index()).mark_bar().encode(
                        x=alt.X('yes:Q', title='Tỉ lệ mở (%)'),
                        y=alt.Y('job:N', sort='-x', title='Nghề'),
                        tooltip=['job:N', alt.Tooltip('yes:Q', format=".2f")]
                    ).properties(width=380, height=300)
                    st.altair_chart(chart)
            with col4:
                st.markdown("#### 👵 Tỉ lệ mở sổ theo độ tuổi")
                if 'age' in df.columns:
                    age_df = df.groupby('age')['deposit'].apply(lambda x: (x == 'yes').mean() * 100).reset_index()
                    line = alt.Chart(age_df).mark_line(point=True).encode(
                        x=alt.X('age:Q', title='Tuổi'),
                        y=alt.Y('deposit:Q', title='Tỉ lệ mở (%)'),
                        tooltip=['age', alt.Tooltip('deposit', format=".2f")]
                    ).properties(width=380, height=300)
                    st.altair_chart(line)
            col5, col6 = st.columns(2, gap="large")
            with col5:
                st.markdown("#### 🥧 Biểu đồ tròn: Tỉ lệ khách mở sổ")
                pie_data = pd.DataFrame({
                    'label': ['Có mở', 'Không mở'],
                    'value': [opened, not_opened]
                })
                pie_option = {
                    "backgroundColor": "#ffffff",
                    "tooltip": {"trigger": "item"},
                    "legend": {
                        "orient": "vertical",
                        "left": "left",
                        "textStyle": {"fontSize": 12}
                    },
                    "series": [{
                        "name": "Tình trạng",
                        "type": "pie",
                        "radius": ["30%", "90%"],
                        "center": ["50%", "50%"],
                        "data": [{"value": row['value'], "name": row['label']} for _, row in pie_data.iterrows()],
                        "label": {
                            "formatter": "{b}: {d}%",
                            "fontSize": 14
                        },
                        "emphasis": {
                            "itemStyle": {
                                "shadowBlur": 10,
                                "shadowOffsetX": 0,
                                "shadowColor": "rgba(0, 0, 0, 0.5)"
                            }
                        }
                    }]
                }
                st_echarts(pie_option, height="710px")

            with col6:
                st.markdown("#### ☎️ Heatmap: Thời lượng & số lần gọi")
                if 'duration' in df.columns and 'campaign' in df.columns:
                    df['duration_grp'] = pd.cut(df['duration'], bins=5)
                    df['campaign_grp'] = pd.cut(df['campaign'], bins=5)
                    heat_df = df.pivot_table(index='duration_grp', columns='campaign_grp',
                                            values='deposit', aggfunc=lambda x: (x == 'yes').mean())
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.heatmap(heat_df, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
                    ax.set_title("Tỉ lệ mở sổ (%)")
                    st.pyplot(fig)
#----------------------------------------
    elif page == "📊 Báo cáo":
        st.subheader("📊 Báo cáo dữ liệu")
        st.write(df.head())

        st.markdown("### Thống kê mô tả")
        st.write(df.describe(include="all"))
        st.markdown("### Tổng quan dữ liệu")
        total = len(df)
        if 'deposit' in df.columns:
            opened = df['deposit'].isin(['yes', 1]).sum()
            not_opened = total - opened
            ratio = (opened / total) * 100 if total > 0 else 0
            st.info(f"""
                - **Tổng khách:** {total}
                - **Có mở sổ:** {opened}
                - **Chưa mở:** {not_opened}
                - **Tỉ lệ mở:** {ratio:.2f}%
            """)
        else:
            st.warning("⛔ File không có cột deposit – không thể phân tích tỷ lệ mở sổ.")

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("#### 📌 Biểu đồ đếm")
            cat_col = st.selectbox("Chọn cột phân loại", cat_cols)
            cat_count = df[cat_col].value_counts().reset_index()
            cat_count.columns = [cat_col, 'Số lượng']
            bar_chart = alt.Chart(cat_count).mark_bar().encode(
                x=alt.X(f'{cat_col}:N', sort='-y', title='Giá trị'),
                y=alt.Y('Số lượng:Q', title='Số lượng'),
                tooltip=[cat_col, 'Số lượng']
            ).properties(width=380, height=300)
            st.altair_chart(bar_chart)
        with col2:
            st.markdown("#### 📌 Histogram (cột số)")
            num_col = st.selectbox("Chọn cột số", num_cols)
            if num_col:
                hist_data = df[[num_col]].dropna()
                hist_chart = alt.Chart(hist_data).mark_bar().encode(
                    alt.X(f"{num_col}:Q", bin=alt.Bin(maxbins=30), title="Giá trị"),
                    y=alt.Y('count()', title="Số lượng"),
                    tooltip=[alt.Tooltip('count()', title="Số lượng")]
                ).properties(width=380, height=300)
                st.altair_chart(hist_chart)
        if 'deposit' in df.columns:
            col3, col4 = st.columns(2, gap="large")
            with col3:
                st.markdown("#### 🧑‍💼 Tỉ lệ mở sổ theo nghề")
                if 'job' in df.columns:
                    job_df = df.groupby('job')['deposit'].value_counts(normalize=True).unstack().fillna(0)*100
                    chart = alt.Chart(job_df.reset_index()).mark_bar().encode(
                        x=alt.X('yes:Q', title='Tỉ lệ mở (%)'),
                        y=alt.Y('job:N', sort='-x', title='Nghề'),
                        tooltip=['job:N', alt.Tooltip('yes:Q', format=".2f")]
                    ).properties(width=380, height=300)
                    st.altair_chart(chart)
            with col4:
                st.markdown("#### 👵 Tỉ lệ mở sổ theo độ tuổi")
                if 'age' in df.columns:
                    age_df = df.groupby('age')['deposit'].apply(lambda x: (x == 'yes').mean() * 100).reset_index()
                    line = alt.Chart(age_df).mark_line(point=True).encode(
                        x=alt.X('age:Q', title='Tuổi'),
                        y=alt.Y('deposit:Q', title='Tỉ lệ mở (%)'),
                        tooltip=['age', alt.Tooltip('deposit', format=".2f")]
                    ).properties(width=380, height=300)
                    st.altair_chart(line)
            col5, col6 = st.columns(2, gap="large")
            with col5:
                st.markdown("#### 🥧 Biểu đồ tròn: Tỉ lệ khách mở sổ")
                pie_data = pd.DataFrame({
                    'label': ['Có mở', 'Không mở'],
                    'value': [opened, not_opened]
                })
                pie_option = {
                    "backgroundColor": "#ffffff",
                    "tooltip": {"trigger": "item"},
                    "legend": {
                        "orient": "vertical",
                        "left": "left",
                        "textStyle": {"fontSize": 12}
                    },
                    "series": [{
                        "name": "Tình trạng",
                        "type": "pie",
                        "radius": ["30%", "90%"],
                        "center": ["50%", "50%"],
                        "data": [{"value": row['value'], "name": row['label']} for _, row in pie_data.iterrows()],
                        "label": {
                            "formatter": "{b}: {d}%",
                            "fontSize": 14
                        },
                        "emphasis": {
                            "itemStyle": {
                                "shadowBlur": 10,
                                "shadowOffsetX": 0,
                                "shadowColor": "rgba(0, 0, 0, 0.5)"
                            }
                        }
                    }]
                }
                st_echarts(pie_option, height="710px")

            with col6:
                st.markdown("#### ☎️ Heatmap: Thời lượng & số lần gọi")
                if 'duration' in df.columns and 'campaign' in df.columns:
                    df['duration_grp'] = pd.cut(df['duration'], bins=5)
                    df['campaign_grp'] = pd.cut(df['campaign'], bins=5)
                    heat_df = df.pivot_table(index='duration_grp', columns='campaign_grp',
                                             values='deposit', aggfunc=lambda x: (x == 'yes').mean())
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.heatmap(heat_df, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
                    ax.set_title("Tỉ lệ mở sổ (%)")
                    st.pyplot(fig)
#--------------------------------------------------------------------
    elif page == "🤖 Dự đoán":
        st.subheader("🤖 Dự đoán mở sổ tiết kiệm")
        if model is None:
            st.warning("Model chưa sẵn sàng!")
        mode = st.radio("Chọn chế độ", ["Dự đoán từng khách", "Dự đoán nhiều khách", "Dự đoán batch"])
        # --- 1. Dự đoán từng khách ---
        if mode == "Dự đoán từng khách":
            inputs = {}
            for col in feature_names:
                display = FIELD_DISPLAY.get(col, col)
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    val = st.slider(display, float(df[col].min()), float(df[col].max()), float(df[col].median()))
                else:
                    val = st.selectbox(display, df[col].dropna().unique().tolist())
                inputs[col] = val
            if st.button("Predict"):
                X_pred = pd.DataFrame([inputs])
                y_pred = model.predict(X_pred)
                result = "✅ Có mở sổ" if y_pred[0] == 1 else "❌ Không mở"
                st.success(f"Kết quả: {result}")
                st.write(X_pred.T.rename(columns={0: "Giá trị"}))
                log_entry = pd.DataFrame([{
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **inputs,
                    "prediction": int(y_pred[0])
                }])
                log_file = "prediction_log.csv"
                if os.path.exists(log_file):
                    log_entry.to_csv(log_file, mode="a", index=False, header=False, encoding="utf-8-sig")
                else:
                    log_entry.to_csv(log_file, index=False, encoding="utf-8-sig")
# Gọi hàm SHAP nếu muốn
                X_pred = X_pred[feature_names]  # trước khi gọi SHAP
                explainer = get_shap_explainer(model, df.sample(100, random_state=42))
                explain_with_shap(explainer ,model, X_pred)
# --- 2. Dự đoán nhiều khách ---
        elif mode == "Dự đoán nhiều khách":
            n_customers = st.slider("Chọn số khách để dự đoán", min_value=5, max_value=100, value=10, step=5)
            randomize = st.checkbox("🔀 Ngẫu nhiên mỗi lần", value=True)
            if st.button("Dự đoán nhóm khách"):
                try:
# Lấy mẫu dữ liệu
                    sample_df = df.sample(n=n_customers) if randomize else df.sample(n=n_customers, random_state=42)
# Lấy đúng input cho model
                    X_input = sample_df[feature_names]
                    # --- Dự đoán ---
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_input)
                        if probs.shape[1] == 2:
                            preds = (probs[:, 1] >= 0.5).astype(int)
                            sample_df["proba_open"] = probs[:, 1]
                        else:
                            preds = model.predict(X_input)
                    else:
                        preds = model.predict(X_input)
                    # Gán kết quả dự đoán
                    sample_df["prediction"] = preds
                    sample_df["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # 🟢 HIỂN THỊ KẾT QUẢ Ở ĐÂY
                    st.subheader("📋 Kết quả dự đoán nhóm khách:")
                    st.dataframe(sample_df)

                    # --- Tạo danh sách cột chuẩn ---
                    base_cols = list(feature_names)
                    if "proba_open" in sample_df.columns:
                        base_cols.append("proba_open")
                    base_cols += ["prediction", "timestamp"]

                    # --- Ghi log ---
                    sample_df[base_cols].to_csv(
                        "prediction_log.csv",
                        mode="a",
                        header=not os.path.exists("prediction_log.csv"),
                        index=False,
                        encoding="utf-8-sig"
                    )
                    st.success("✅ Dự đoán & log nhóm thành công!")
                except Exception as e:
                    st.error(f"❌ Lỗi khi dự đoán nhóm: {e}")
# --- 3. Dự đoán batch ---
        elif mode == "Dự đoán batch":
            file_batch = st.file_uploader("📁 Chọn file batch (.csv hoặc .xlsx):", type=["csv", "xlsx"])
            if file_batch:
                try:
                    if file_batch.name.endswith('.csv'):
                        df_batch = pd.read_csv(file_batch)
                    else:
                        df_batch = pd.read_excel(file_batch)

                    preds = model.predict(df_batch[feature_names])
                    df_batch["Dự đoán"] = ["Có mở" if p == 1 else "Không mở" for p in preds]
                    st.dataframe(df_batch)

                    csv = df_batch.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 Tải kết quả", data=csv, file_name="ketqua_batch.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Lỗi: {e}")
        else:
            file_batch = st.file_uploader(
                "📁 Chọn file batch (.csv hoặc .xlsx):",
                type=["csv", "xlsx"],
                accept_multiple_files=False
            )
            if file_batch:
                try:
                    if file_batch.name.endswith('.csv'):
                        df_batch = pd.read_csv(file_batch)
                    elif file_batch.name.endswith('.xlsx'):
                        df_batch = pd.read_excel(file_batch)

                    preds = model.predict(df_batch)
                    df_batch['Dự đoán'] = ["Có mở" if p==1 else "Không mở" for p in preds]
                    st.dataframe(df_batch)
                    csv = df_batch.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 Tải kết quả", data=csv, file_name="ketqua_batch.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Lỗi: {e}")
    elif page == "🔬 So sánh":
        st.subheader("🔬 So sánh thực tế vs dự đoán")

        file_true = st.file_uploader("Upload file thực tế (có cột deposit)", type=["csv"])
        file_pred = st.file_uploader("Upload file batch đã dự đoán", type=["csv"])

        if file_true and file_pred:
            try:
                df_true = pd.read_csv(file_true)
                df_pred = pd.read_csv(file_pred)

                df_true['deposit'] = df_true['deposit'].replace({1: 'yes', 0: 'no'})
                df_pred['Dự đoán'] = df_pred['Dự đoán'].replace({'Có mở': 'yes', 'Không mở': 'no'})

                compare = pd.concat([df_true.reset_index(drop=True), df_pred['Dự đoán']], axis=1)
                compare['So sánh'] = compare.apply(lambda row: "✅" if row['deposit'] == row['Dự đoán'] else "❌", axis=1)
                acc = (compare['deposit'] == compare['Dự đoán']).mean() * 100

                st.success(f"🎯 Độ chính xác: {acc:.2f}%")
                st.dataframe(compare)

                csv_out = compare.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button("📥 Tải kết quả so sánh", data=csv_out, file_name="so_sanh.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Lỗi xử lý: {e}")
    elif page == "📈 Theo dõi mô hình":
            monitor.render_log_page()
# 🧩 Phân tích EDA tự động--------------------        
    elif page == "🧩 Phân tích EDA tự động":
        st.subheader("🧩 Auto-EDA với YData Profiling")
        if df is not None:
            eda_tool = st.selectbox("Chọn công cụ", ["YData Profiling", "Sweetviz"])
            if st.button("🚀 Phân tích dữ liệu"):
                if eda_tool == "YData Profiling":
                    run_ydata_profiling(df)
                else:
                    run_sweetviz(df)
        else:
            st.warning("⛔ Bạn cần nạp dữ liệu trước để phân tích.")

    elif page == "📈 Model comparison":
        # Ép y_train và y_test thành int nếu đang ở dạng float
        st.session_state.y_train = st.session_state.y_train.astype(int)
        st.session_state.y_test = st.session_state.y_test.astype(int)
        model_compare.render_comparison_page()

if __name__ == '__main__':
    main()