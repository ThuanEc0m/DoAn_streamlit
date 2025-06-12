import streamlit as st
from streamlit_echarts import st_echarts
import pandas as pd
import joblib

# 1. Cấu trúc thư mục (nếu không dùng uploader)
#    .
#    ├── streamlit_app.py
#    ├── requirements.txt
#    ├── model.pkl

# 2. Hàm load dữ liệu linh hoạt (CSV hoặc Excel)
@st.cache_data
def load_data(file) -> pd.DataFrame:
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    else:
        st.error('Chỉ hỗ trợ file .csv, .xls, .xlsx')
        return pd.DataFrame()

@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Không tải được model: {e}")
        return None

# 3. Vẽ chart ECharts động

def draw_chart(df: pd.DataFrame):
    st.subheader("Chọn kiểu chart")
    chart_type = st.selectbox('Loại biểu đồ', ['Bar Chart (Đếm)', 'Histogram'])
    if chart_type == 'Bar Chart (Đếm)':
        col = st.selectbox('Chọn cột (categorical)', df.select_dtypes(include=['object', 'category']).columns.tolist())
        data = df[col].value_counts().reset_index()
        data.columns = [col, 'count']
        option = {
            'title': {'text': f'Biểu đồ đếm {col}'},
            'tooltip': {'trigger': 'axis'},
            'xAxis': {'type': 'category', 'data': data[col].tolist()},
            'yAxis': {'type': 'value'},
            'series': [{'type': 'bar', 'data': data['count'].tolist(), 'name': 'Count'}]
        }
    else:
        num_col = st.selectbox('Chọn cột (numeric)', df.select_dtypes(include=['int64', 'float64']).columns.tolist())
        bins = st.slider('Số bins', 5, 50, 20)
        values = df[num_col].dropna().tolist()
        option = {
            'title': {'text': f'Histogram của {num_col}'},
            'tooltip': {},
            'xAxis': {'type': 'value', 'splitNumber': bins},
            'yAxis': {'type': 'value'},
            'series': [{'type': 'histogram', 'data': values, 'name': num_col}]
        }
    st_echarts(options=option, height='450px')

# 4. Giao diện chính

def main():
    st.title('Ứng dụng Phân tích & Dự đoán')

    # Upload file
    st.sidebar.header('1. Upload dữ liệu')
    data_file = st.sidebar.file_uploader('Chọn file CSV/Excel', type=['csv', 'xls', 'xlsx'])
    if data_file:
        df = load_data(data_file)
    else:
        st.info('Vui lòng upload file dữ liệu để tiếp tục')
        return

    # Load model
    st.sidebar.header('2. Tải mô hình')
    model = load_model('model.pkl')

    # Chọn trang
    st.sidebar.header('3. Chọn chức năng')
    page = st.sidebar.selectbox('Trang', ['Báo cáo', 'Dự đoán'])

    if page == 'Báo cáo':
        st.header('📊 Báo cáo dữ liệu')
        st.write('Kích thước:', df.shape)
        st.dataframe(df.head(10))
        draw_chart(df)

    else:
        st.header('🤖 Dự đoán mở term deposit')
        if model is None:
            st.warning('Chưa có mô hình. Vui lòng kiểm tra model.pkl')
            return
        # Tạo input tự động theo cột numeric
        inputs = {}
        st.subheader('Thông tin khách hàng')
        cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        for c in cols:
            min_val, max_val = int(df[c].min()), int(df[c].max())
            default = int(df[c].median())
            inputs[c] = st.slider(c, min_val, max_val, default)

        # Dự đoán
        if st.button('Predict'):
            X_pred = pd.DataFrame([inputs])
            try:
                y = model.predict(X_pred)
                st.success(f'Kết quả dự đoán: {y[0]}')
            except Exception as e:
                st.error(f'Lỗi khi dự đoán: {e}')

if __name__ == '__main__':
    main()