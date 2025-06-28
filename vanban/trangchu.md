
------------------------------------------------------------------------------


# 🏠 Trang chủ
## 🎯 Giới thiệu dự án

Dự án phân tích hành vi và đặc điểm của khách hàng nhằm xác định nhóm khách hàng tiềm năng có khả năng cao mở **sổ tiết kiệm có kỳ hạn**. Thông qua các kỹ thuật phân tích dữ liệu và mô hình dự đoán, dự án hỗ trợ ngân hàng nâng cao hiệu quả marketing và chăm sóc khách hàng.
 Với các khoản tiền gửi có kỳ hạn nổi lên như một công cụ giảm thiểu rủi ro hiệu quả. Trong lĩnh vực tiếp thị, các phương pháp giao tiếp trực tiếp đã thay thế các trung gian truyền thống. Tận dụng dữ liệu từ một ngân hàng , nghiên cứu này phân tích độ chính xác của các dự đoán về những người tiêu dùng có khả năng cao sẽ gửi tiền có kỳ hạn. Bằng cách sử dụng các thuật toán học máy, bao gồm Support Vector Machine (SVM), Gaussian Naïve Bayes, Decision Tree, Bagging, Light Gradient Boosting (GBM), thuật toán bagging và Extreme Gradient Boosting (XgBoost), để phân loại và dự đoán khách hàng tiềm năng. Các phát hiện làm nổi bật hiệu suất tối ưu của các thuật toán bagging trong việc dự đoán hành vi của khách hàng, tăng cường các chiến lược tiếp thị.
## 🧾 Nguồn dữ liệu
Dữ liệu bao gồm:

- Kích thước:
Tổng số bản ghi (rows): 11.162 khách hàng
Tổng số thuộc tính (features/columns): 17 thuộc tính (trong đó có 16 biến đầu vào và 1 biến mục tiêu)
-	age – Tuổi khách hàng
Phạm vi: từ khoảng 18 đến trên 90
-	job – Nghề nghiệp hiện tại
-	marital – Tình trạng hôn nhân
-	education – Trình độ học vấn
-	default – Tình trạng vỡ nợ tín dụng trước đó
-	balance – Số dư tài khoản (số tiền đang có)
-	housing – Khách hàng có đang vay mua nhà hay không
-	loan – Khách hàng có đang vay tiêu dùng (personal loan) hay không
-	contact – Phương thức liên lạc cuối cùng (đợt gọi điện gần nhất)
-	day – Ngày trong tháng khi liên hệ cuối cùng
-	month – Tháng khi liên hệ cuối cùng
-	duration – Thời lượng đàm thoại lần liên hệ cuối cùng (tính bằng giây)
-	campaign – Số lần liên hệ đã thực hiện trong chiến dịch hiện tại (campaign) với khách hàng
-	pdays – Số ngày kể từ lần liên hệ cuối cùng trong chiến dịch cũ với khách hàng (previous campaign)
-	previous – Số lần liên hệ đã thực hiện trong chiến dịch cũ
-	poutcome – Kết quả của chiến dịch tiếp cận trước (previous campaign outcome)
-	deposit – Biến mục tiêu (target variable): khách hàng có mở sổ tiết kiệm có kỳ hạn hay không


Thời gian thu thập: *01/01/2023 – 31/12/2023*

## 🔍 Phương pháp phân tích
- Làm sạch & tiền xử lý dữ liệu
- Khám phá dữ liệu (EDA)
- Phân tích mối quan hệ giữa các biến
- Xây dựng mô hình phân loại khách hàng (Random Forest, XGBoost,...)
- Đánh giá hiệu quả mô hình

## 📊 Các tính năng chính
- Thống kê & biểu đồ tương tác theo độ tuổi, giới tính, thu nhập,...
- Phân nhóm khách hàng theo hành vi
- Dự đoán khả năng mở sổ tiết kiệm
- Gợi ý nhóm khách hàng tiềm năng cho chiến dịch marketing

## 🛠️ Công nghệ sử dụng
- **Ngôn ngữ**: Python
- **Thư viện**: Pandas, Scikit-learn, Streamlit, Matplotlib, Seaborn, Altair
- **Giao diện**: Streamlit thân thiện và dễ sử dụng

## 👥 Đối tượng sử dụng
- Phòng Marketing
- Phòng chăm sóc khách hàng
- Quản lý chiến dịch kinh doanh

## 📌 Kết quả nổi bật
- Khách hàng từ 25–40 tuổi, có thu nhập trung bình cao, thường xuyên giao dịch có xu hướng mở sổ tiết kiệm nhiều hơn.
- Mô hình dự đoán đạt **accuracy ~85%**, hỗ trợ sàng lọc khách hàng hiệu quả.

## 📬 Liên hệ
Nhóm phân tích dữ liệu – Dự án ngân hàng 2025  
📧 Email: ngockisune@example.com

## Lưu ý:
Dữ liệu cần được làm sạch trước khi đưa vào.