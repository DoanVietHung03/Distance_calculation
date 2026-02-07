- Dự án sử dụng công nghệ Monocular Depth Estimation (Ước lượng độ sâu từ một camera) kết hợp với Deep Learning để đo khoảng cách trong không gian 3D. Hệ thống không cần camera đôi (Stereo) hay cảm biến LiDAR, mà sử dụng mô hình AI để dự đoán bản đồ độ sâu (Depth Map) và tái tạo tọa độ 3D thực tế dựa trên mô hình Camera Pinhole.

1) Tính Năng Chính:
- AI Depth Perception: Sử dụng mô hình Depth-Anything-V2 (SOTA) để tạo ra bản đồ độ sâu chi tiết từ ảnh 2D, cho phép hiểu được cấu trúc xa gần của không gian.
- Auto-Calibration (Scale Factor): Tự động tính toán hệ số tỷ lệ (Scale Factor) thông qua đối tượng tham chiếu (người) được phát hiện bởi YOLOv11. Giúp chuyển đổi giá trị độ sâu trừu tượng của AI sang đơn vị mét.
- 3D Reconstruction: Tái tạo tọa độ (X, Y, Z) của bất kỳ điểm pixel nào trong ảnh dựa trên mô hình máy ảnh lỗ kim (Pinhole Camera Model).
- Flexible Measurement: Khác với phương pháp Homography (chỉ đo trên mặt sàn), phương pháp này cho phép đo khoảng cách giữa 2 điểm bất kỳ trong không gian 3D (ví dụ: từ tay người này sang đầu người kia, hoặc khoảng cách từ camera đến vật thể).

2) Phương Pháp & Nguyên Lý Hoạt Động:
a) Giai đoạn 1: Hiệu Chuẩn Tỷ Lệ (Scale Factor Calibration) - scale_factor_cal.py:
- Mô hình AI trả về "độ sâu tương đối" (relative depth), không có đơn vị mét. Bước này dùng để tìm ra hệ số quy đổi.
- Bước 1. Object Detection & Depth Inference:
    + Sử dụng YOLOv11 để tìm người trong ảnh.
    + Đồng thời chạy mô hình Depth Anything để lấy bản đồ độ sâu (Raw Depth Map).
- Bước 2. Tham Chiếu Thực Tế:
    + Hệ thống lấy độ sâu tại tâm bounding box của người.
    + Người dùng cung cấp khoảng cách thực tế từ camera đến người đó (ví dụ: 6.0 mét).
- Bước 3. Tính Hệ Số Scale:
    + Công thức áp dụng: Scale_Factor = Khoảng_Cách_Thực * Giá_Trị_Raw_Depth (Do mô hình Depth Anything thường trả về giá trị nghịch đảo của độ sâu - Inverse Depth).
    + Kết quả là một con số (ví dụ: 12.82) dùng để cấu hình cho file đo lường.
b) Giai đoạn 2: Đo Đạc Khoảng Cách 3D (3D Measurement) - distance_cal.py:
- Sau khi có SCALE_FACTOR, hệ thống có thể đo khoảng cách bất kỳ.
- Bước 1. Tạo Bản Đồ Độ Sâu Mét (Metric Depth Map):Từ ảnh đầu vào, AI dự đoán Raw Depth.Tính độ sâu Z (mét) cho từng pixel: $Z = \frac{SCALE\_FACTOR}{Raw\_Depth}$.
- Bước 2. Back-Projection (Chiếu Ngược 2D sang 3D):Sử dụng mô hình Pinhole Camera để chuyển đổi tọa độ pixel $(u, v)$ và độ sâu $Z$ thành tọa độ không gian thực $(X, Y, Z)$.Công thức:$$X = \frac{(u - c_x) \cdot Z}{f_x}$$$$Y = \frac{(v - c_y) \cdot Z}{f_y}$$Trong đó: $(c_x, c_y)$ là tâm ảnh, $(f_x, f_y)$ là tiêu cự (Focal Length). Hệ thống tự động ước lượng tiêu cự nếu không có thông số camera matrix.
- Bước 3. Tính Khoảng Cách Euclidean:Khi người dùng click 2 điểm trên ảnh, hệ thống lấy tọa độ 3D của chúng ($P_1$ và $P_2$).Khoảng cách được tính toán: $Distance = \sqrt{(X_2-X_1)^2 + (Y_2-Y_1)^2 + (Z_2-Z_1)^2}$.

3) Lưu Ý Kỹ Thuật:
- Cấu hình phần cứng: Depth-Anything-V2 là một mô hình Transformer khá nặng, khuyến nghị chạy trên GPU (NVIDIA CUDA) để đạt tốc độ xử lý tốt nhất. Nếu chạy CPU sẽ có độ trễ khi click chuột.
- Độ chính xác của Tiêu cự (Focal Length): File distance_cal.py đang sử dụng cơ chế ước lượng tiêu cự (FOCAL_LENGTH = Width * 0.9 hoặc 1600). Để đo chính xác tuyệt đối, cần nhập đúng tiêu cự thực của camera (có thể lấy từ thông số kỹ thuật hoặc qua quá trình calibrate bàn cờ vua).
- Ánh sáng & Phản chiếu: Ảnh quá tối hoặc bề mặt phản chiếu mạnh (gương, kính) có thể làm mô hình Depth dự đoán sai độ sâu, dẫn đến kết quả đo không chính xác.
- Inverse Depth: Lưu ý rằng giá trị Raw từ model tỉ lệ nghịch với khoảng cách thực (Raw càng lớn thì vật càng gần, Raw càng nhỏ thì vật càng xa).

4) Yêu Cầu Hệ Thống:
- Python 3.10+
- Thư viện chính: torch, opencv-python, transformers, ultralytics (cho YOLO), pillow.
- GPU NVIDIA (VRAM >= 4GB) để chạy mượt mà mô hình Depth-Anything-Small.