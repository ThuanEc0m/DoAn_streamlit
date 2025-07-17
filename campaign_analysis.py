# campaign_analysis.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def render_campaign_analysis():
    st.header("📉 Phân tích hiệu quả chiến dịch mở sổ tiết kiệm")

    # Kiểm tra dữ liệu đã sẵn sàng chưa
    if "df" not in st.session_state:
        st.warning("⚠️ Vui lòng nạp dữ liệu trước.")
        return

    df = st.session_state.df.copy()
    df.columns = df.columns.str.lower()

    if "month" not in df.columns or "deposit" not in df.columns:
        st.error("❌ Thiếu cột 'month' hoặc 'deposit' trong dữ liệu.")
        return

    # Tỷ lệ chuyển đổi - đưa lên đầu
    total_customers = len(df)
    converted = (df.deposit == "yes").sum()
    conversion_rate = round(converted / total_customers * 100, 2)
    st.success(f"🎯 Tỷ lệ chuyển đổi tổng thể: {conversion_rate}%")

    # Danh sách tháng chuẩn và ánh xạ hiển thị
    month_order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    month_map = {
        'jan': 'Jan', 'feb': 'Feb', 'mar': 'Mar', 'apr': 'Apr',
        'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'aug': 'Aug',
        'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dec': 'Dec'
    }

    # Biểu đồ 1: Số lượng khách mở sổ theo tháng
    st.subheader("📅 Số lượng khách mở sổ theo tháng")
    with st.expander("📌 Tuỳ chọn lọc theo tháng"):
        selected_months = st.multiselect("Chọn tháng", options=sorted(df["month"].unique()), default=sorted(df["month"].unique()))
    df_month = df[df["month"].isin(selected_months)]

    df_month["month"] = df_month["month"].str.lower()
    month_count_raw = df_month[df_month.deposit == "yes"].month.value_counts()
    month_count = pd.Series({month: month_count_raw.get(month, 0) for month in month_order})
    month_count.index = [month_map[m] for m in month_order]

    fig1, ax1 = plt.subplots(figsize=(7,4))
    bars = ax1.bar(month_count.index, month_count.values, color="#2b7bba")
    ax1.set_title("Số lượng khách mở sổ theo tháng", fontsize=13, weight="bold")
    ax1.set_xlabel("Tháng")
    ax1.set_ylabel("Số lượng")
    ax1.bar_label(bars, padding=3, fontsize=9)
    st.pyplot(fig1)

    # Biểu đồ 2: Tổng số dư theo tháng
    st.subheader("💰 Tổng số dư tiền gửi theo tháng")
    balance_by_month_raw = df_month[df_month.deposit == "yes"].groupby("month")["balance"].sum()
    balance_by_month = pd.Series({month: balance_by_month_raw.get(month, 0) for month in month_order})
    balance_by_month.index = [month_map[m] for m in month_order]

    fig2, ax2 = plt.subplots(figsize=(7,4))
    ax2.plot(balance_by_month.index, balance_by_month.values, marker='o', color="#4CAF50", linewidth=2)
    for i, val in enumerate(balance_by_month.values):
        ax2.text(i, val + 500, f"{int(val):,}", ha='center', fontsize=8)
    ax2.set_title("Tổng số dư gửi mới theo tháng", fontsize=13, weight="bold")
    ax2.set_xlabel("Tháng")
    ax2.set_ylabel("Tổng số dư (EUR)")
    st.pyplot(fig2)

    # Biểu đồ 3: Top nghề
    st.subheader("👥 Top 10 nghề có khách hàng mở sổ nhiều nhất")
    with st.expander("📌 Tuỳ chọn lọc theo nghề"):
        selected_jobs = st.multiselect("Chọn nghề", options=df["job"].unique(), default=list(df["job"].unique()))
    df_job = df[df["job"].isin(selected_jobs)]

    job_count = df_job[df_job.deposit == "yes"].job.value_counts().head(10)

    fig3, ax3 = plt.subplots(figsize=(7,4))
    sns.barplot(y=job_count.index, x=job_count.values, palette="crest", ax=ax3)
    ax3.set_title("Top 10 nghề có số lượng mở sổ cao", fontsize=13, weight="bold")
    ax3.set_xlabel("Số khách mở sổ")
    ax3.set_ylabel("Nghề")
    for i, v in enumerate(job_count.values):
        ax3.text(v + 5, i, str(v), color='black', va='center', fontsize=9)
    st.pyplot(fig3)