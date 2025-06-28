import streamlit as st
import json
import os
import base64
import datetime
from streamlit_lottie import st_lottie

def render_sidebar_lottie(json_path="assets/pew.json", height=170):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            lottie_data = json.load(f)
        with st.sidebar:
            st.markdown("""
                <div style="border-radius: 20px; overflow: hidden; width: 100%;">
                    <div id="lottie-container"></div>
                </div>
            """, unsafe_allow_html=True)
            st_lottie(lottie_data, height=height, key="sidebar_lottie", loop=True, speed=1)
    else:
        st.sidebar.warning("⚠️ Không tìm thấy animation JSON.")



# ----------- Hàm render tiêu đề có logo ----------- 
def render_title_with_image(image_path, title_text):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(f"""
        <div style="display:flex; align-items:center; margin-top:-30px; margin-bottom:20px;">
            <img src="data:image/png;base64,{encoded}" style="width:100px; height:100px; margin-right:15px;">
            <h1 style="font-size: 300%;">{title_text}</h1>
        </div>
    """, unsafe_allow_html=True)

# ----------- Giao diện tổng thể ----------- 
def set_custom_theme():
    st.markdown("""
        <style>
        [data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(180deg, #90CAF9, #66b2ff);
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
        }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(to bottom, #ffffff 0%, #B3E5FC 100%);
            padding: 20px;
        }
        body { background-color: #f0f8ff; }
        .stSelectbox div[data-baseweb="select"] > div {
            font-weight: 600;
            color: #333333;
        }
        </style>
    """, unsafe_allow_html=True)

# ----------- Hiển thị thời gian góc phải ----------- 
def render_time_top_right():
    now = datetime.datetime.now()
    time_str = now.strftime("%A, %d/%m/%Y %H:%M")
    st.markdown(
        f"<div style='text-align:right; color:gray; font-size:14px; margin-bottom:10px;'>⏰ {time_str}</div>",
        unsafe_allow_html=True
    )

def render_data_image():
    st.sidebar.image("assets/data.png", use_container_width=True)

def render_logo_bottom():
    st.sidebar.markdown(
        """
        <div style="position: absolute; bottom: 10px; width: 100%; text-align: center;">
            <img src="https://i.imgur.com/YOUR_IMAGE.png" style="max-width: 80%;">
        </div>
        """,
        unsafe_allow_html=True
    )
