import streamlit as st
import pandas as pd
import os
import altair as alt

def render_log_page():
    st.subheader("📈 Theo dõi mô hình")
    log_file = "prediction_log.csv"
    if not os.path.exists(log_file):
        st.info("⛔ Chưa có log nào được ghi.")
        return
    # --- Xoá toàn bộ log ---
    if os.path.exists(log_file):
        if st.button("🗑️ Xoá toàn bộ dữ liệu log"):
            os.remove(log_file)
            st.success("Đã xoá log.")
            return

    # --- Đọc log ---
    try:
        df_log = pd.read_csv(log_file, on_bad_lines='skip')
    except Exception as e:
        st.error(f"Không đọc được log: {e}")
        return

    if df_log.empty:
        st.info("⛔ Chưa có dữ liệu log.")
        return

    # --- Chọn chế độ hiển thị ---
    use_all = st.checkbox("📋 Dùng toàn bộ log", value=True)

    if use_all:
        df_view = df_log.copy()
    else:
        n_rows = st.slider("Số dòng log gần nhất muốn xem", min_value=10, max_value=min(2000, len(df_log)), value=50, step=10)
        df_view = df_log.tail(n_rows)

    # --- Hiển thị bảng ---
    st.markdown("### 📋 Dữ liệu log")
    st.dataframe(df_view)

    # --- Biểu đồ tỉ lệ ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Tỉ lệ dự đoán mở sổ")
        if "prediction" in df_view.columns:
            pred_ratio = df_view["prediction"].value_counts(normalize=True) * 100
            chart_data = pd.DataFrame({
                "Trạng thái": ["Không mở", "Có mở"],
                "Tỉ lệ": [pred_ratio.get(0, 0), pred_ratio.get(1, 0)]
            })
            st.bar_chart(chart_data.set_index("Trạng thái"))
        else:
            st.warning("Không tìm thấy cột 'prediction'.")

    with col2:
        st.markdown("### 📆 Tỉ lệ mở sổ theo tuần")
        if "timestamp" in df_view.columns:
            try:
                df_view['timestamp'] = pd.to_datetime(df_view['timestamp'], errors='coerce')
                df_view = df_view.dropna(subset=['timestamp'])

                weekly_acc = df_view.groupby(pd.Grouper(key='timestamp', freq='W'))['prediction'].mean().reset_index()
                weekly_acc['accuracy (%)'] = weekly_acc['prediction'] * 100

                chart = alt.Chart(weekly_acc).mark_line(point=True).encode(
                    x='timestamp:T',
                    y='accuracy (%):Q',
                    tooltip=['timestamp:T', 'accuracy (%):Q']
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)
            except:
                st.warning("⛔ Không thể hiển thị biểu đồ thời gian.")
