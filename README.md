# BTL_TIMESERIES
# 🌤️ Dự Báo Bức Xạ Mặt Trời Từ Dữ Liệu Bức Xạ Và Ảnh Sky Cam

## 📌 Giới thiệu

Đây là bài tập lớn với mục tiêu xây dựng hệ thống **dự báo bức xạ mặt trời ngắn hạn**, kết hợp giữa:

- **Chuỗi thời gian dữ liệu bức xạ** 
- **Ảnh Sky Cam** (ảnh chụp bầu trời thời gian thực)
- **Các dữ liệu được thu thập sau mỗi 10s**

Nhóm sử dụng các mô hình kết hợp như: **Resnet+LSTM**, **Resnet+Transformer** và **Resnet+Timeformers** được sử dụng để nâng cao độ chính xác.
Các mô hình dự đoán trong thời gian ngắn là 5 phút

## 🧠 Kiến trúc và kết quả của từng mô hình được trình bày trong thư mục report có tên file là: BTL_TS.docx

## 📂 Cấu trúc thư mục
**BTL_TIMESERIES/**
- ├── data/ # Dữ liệu ảnh & bức xạ 
- ├── CodeModel/ # Phần xây dựng và huấn luyện mô hình
- ├── Best_Model/ # Lưu kết quả của các mô hình
- ├── reports/ # Báo cáo
- ├── requirements.txt # Danh sách thư viện
- └── README.md

## 🚀 Cách chạy mô hình

- Truy cập vào thư mục Codemodels
- Tải dữ liệu và mô hình muốn chạy về 
- Sử dụng trên kaggle hoặc môi trường bất kỳ (ưu tiên trên kaggle vì đã tải sẵn dữ liệu và có GPU để huấn luyện mô hình)
- Chạy mô hình
