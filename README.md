# 🩺 Ứng Dụng Dự Đoán Nguy Cơ Tiểu Đường bằng bộ 21 câu hỏi

### 📋 Khảo sát 21 câu hỏi – Ứng dụng Machine Learning
### 🌐 Triển khai dưới dạng Web App tương tác

## 📌 Giới Thiệu Dự Án

- Đây là một ứng dụng Machine Learning dùng để dự đoán nguy cơ mắc bệnh tiểu đường dựa trên bộ câu hỏi khảo sát gồm 21 câu hỏi liên quan đến sức khỏe và lối sống.

- Khác với xét nghiệm y khoa truyền thống, hệ thống này ước tính nguy cơ thông qua:

🧠 Mô hình Machine Learning
📊 Dữ liệu khảo sát sức khỏe
🌐 Giao diện Web tương tác
📈 Dự đoán theo xác suất (%)
🎯 Mục Tiêu Dự Án

✔ Xây dựng mô hình phân loại nguy cơ tiểu đường
✔ Sử dụng dữ liệu khảo sát thay vì dữ liệu xét nghiệm
✔ Triển khai mô hình thành Web App thời gian thực
✔ Hiển thị xác suất dự đoán rõ ràng
✔ Thiết kế giao diện thân thiện, dễ sử dụng

## Giới thiệu về bộ dữ liệu 
Bộ dữ liệu là file csv 253k lấy trên kaggle có 21 cột "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"
- Được chuyển đổi thành các câu hỏi để người dùng dễ dàng trả lời
## 🧪 Các Nhóm Câu Hỏi Khảo Sát
### 🧍 Thông Tin Cá Nhân

BMI (hoặc tự động tính từ chiều cao & cân nặng)
Tuổi
Giới tính
Trình độ học vấn
Thu nhập

### ❤️ Tiền Sử Bệnh

Cao huyết áp
Cholesterol cao
Đột quỵ
Bệnh tim
Đã kiểm tra cholesterol trong 5 năm

### 🏃 Lối Sống

Hút thuốc
Hoạt động thể chất
Ăn trái cây
Ăn rau xanh
Uống rượu nhiều

### 🏥 Tiếp Cận Y Tế

Có bảo hiểm y tế
Không đi khám do chi phí

### 📊 Tình Trạng Sức Khỏe Gần Đây

Số ngày sức khỏe tinh thần kém
Số ngày sức khỏe thể chất kém
Khó khăn khi đi lại
Đánh giá sức khỏe tổng quát

### 🧠 Mô Hình Machine Learning

Hệ thống sử dụng mô hình phân loại để dự đoán:

Xác suất nguy cơ mắc tiểu đường (%)

Ngưỡng phân loại (threshold) 
Mặc định: 0.65

## ⚙ Quy Trình Xử Lý

#### 1️⃣ Tiền xử lý dữ liệu
#### 2️⃣ Chuẩn hóa và mã hóa đặc trưng
#### 3️⃣ Huấn luyện mô hình
#### 4️⃣ Lưu mô hình bằng Joblib
#### 5️⃣ Dự đoán xác suất khi người dùng gửi khảo sát


## 🖥️ Giao Diện Web

Ứng dụng được xây dựng bằng:

🐍 Python
🌐 Flask
📊 Scikit-learn
🎨 HTML / CSS
✨ Tính Năng Giao Diện

- Giao diện card hiện đại
- Thanh tiến trình khảo sát
- Hiển thị xác suất bằng thanh progress
- Trang kết quả rõ ràng
- Responsive (tương thích nhiều màn hình)

## 📂 Cấu Trúc Thư Mục
#### ML_Diabetes_Risk/
#### │
#### ├── app.py                 # Backend Flask
#### ├── model.joblib           # Mô hình ML đã huấn luyện
#### ├── templates/
#### │   ├── index.html         # Trang khảo sát
#### │   └── result.html        # Trang kết quả
#### ├── static/                # Tài nguyên 
#### └── README.md

## 🚀 Hướng Dẫn Chạy Dự Án
#### 1️⃣ Clone Repository
git clone https://github.com/ten-cua-ban/ten-repo.git
cd ten-repo
#### 2️⃣ Cài Đặt Thư Viện
pip install -r requirements.txt
#### 3️⃣ Chạy Ứng Dụng
python app.py

Mở trình duyệt tại:
http://127.0.0.1:5000

### 📊 Kết Quả Trả Về

🟢 Không nguy cơ
🔴 Có nguy cơ

📈 Xác suất phần trăm cụ thể (ví dụ: 55.23%)

# ⚠ Lưu ý: Đây chỉ là công cụ sàng lọc, không thay thế chẩn đoán y khoa chuyên nghiệp.

## 🛠️ Công Nghệ Sử Dụng
Công Nghệ	Vai Trò
Python 🐍	Backend & ML
Flask 🌐	Web Framework
Scikit-learn 📊	Huấn luyện mô hình
Joblib 📦	Lưu & load model
HTML/CSS 🎨	Thiết kế giao diện
📈 Hướng Phát Triển Tương Lai

🔄 So sánh nhiều mô hình (Random Forest, XGBoost…)

📊 Thêm biểu đồ Confusion Matrix

🌍 Deploy lên Render / Railway / AWS

🔐 Thêm hệ thống đăng nhập

🗄 Lưu lịch sử dự đoán vào Database

## 👨‍💻 Tác Giả

Le Tran Duy
AI & Machine Learning Developer

⭐ Nếu bạn thấy dự án hữu ích
Hãy cho một ⭐ trên GitHub để ủng hộ nhé!
