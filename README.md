# Road Segmentation GUI Tool

Một ứng dụng Desktop GUI mạnh mẽ được xây dựng bằng Python (Tkinter) để thực hiện phân đoạn đường (Road Segmentation) từ ảnh vệ tinh. Công cụ này tích hợp nhiều kiến trúc Deep Learning tiên tiến và cho phép tinh chỉnh hình ảnh đầu vào theo thời gian thực.

## ✨ Tính năng chính

* **Đa dạng Mô hình:** Hỗ trợ nhiều kiến trúc Segmentation hàng đầu:
    * DeepLabV3+
    * EfficientViT-Seg
    * FPN, MAnet, PAN, PSPNet, UPerNet
* **Backbone linh hoạt:** Hỗ trợ các encoder như `mobileone_s0` -> `s3`, `efficientvit-seg`, v.v.
* **Xử lý ảnh (Pre-processing):** Tích hợp công cụ OpenCV cho phép:
    * Làm mờ (Blur)
    * Điều chỉnh độ sáng (Brightness)
    * Thêm nhiễu Gaussian (Noise) để kiểm thử độ bền vững của mô hình.
* **Chế độ chạy:**
    * Xử lý từng ảnh (Single Image).
    * Xử lý hàng loạt cả thư mục (Batch Processing).
* **Giao diện trực quan:** Xem trước ảnh gốc, kết quả chồng lớp (Overlay) và mặt nạ (Mask) song song.

## 🛠️ Cài đặt

1.  **Clone repository:**
    ```bash
    git clone [https://github.com/username-cua-ban/road-segmentation-gui.git](https://github.com/username-cua-ban/road-segmentation-gui.git)
    cd road-segmentation-gui
    ```

2.  **Tạo môi trường Conda (Khuyên dùng):**
    ```bash
    conda create -n Map python=3.10
    conda activate Map
    ```

3.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Hướng dẫn sử dụng

### Chạy ứng dụng
* **Linux/Mac:** Chạy file `run.sh` hoặc lệnh:
    ```bash
    python UI_segmentation.py
    ```

### Các bước thực hiện trên GUI:
1.  **Select Input:** Chọn ảnh lẻ hoặc folder ảnh cần dự đoán.
2.  **Image Processing:** Kéo thanh trượt để chỉnh sửa ảnh (nếu cần test nhiễu).
3.  **Model Configuration:**
    * Chọn Architecture (ví dụ: DeepLabV3Plus).
    * Chọn Encoder và Dataset.
    * Chọn file weights (`.pt`).
    * Nhấn **Load Model**.
4.  **Run:** Nhấn nút **RUN DETECT** và chờ kết quả.

## 📂 Cấu trúc thư mục

* `UI_segmentation`: File chính khởi chạy giao diện người dùng.
* `inference.py`: Core xử lý logic, load model và dự đoán.
* `gui_config.py`: Quản lý đường dẫn và cấu hình các model.
* `weight_data/`: Thư mục chứa các file weights (Lưu ý: Bạn cần tự tải weights về đúng cấu trúc).

## ⚠️ Lưu ý về Model Weights

Do kích thước file weights (`.pt`) thường lớn, chúng không được upload trực tiếp lên GitHub này. Vui lòng tải weights và đặt vào thư mục `weight_data/` theo cấu trúc được định nghĩa trong `gui_config.py`.

Không có thể tải về những weight có sẵn: [weight_data](https://drive.google.com/drive/folders/1Xo9MOrquM-1DjhHSwdEEOqw-q1Iee1i7)

## 🤝 Đóng góp

Mọi đóng góp (Pull Request) hoặc báo lỗi (Issue) đều được hoan nghênh.

---
Developed for Road Segmentation Research.
