from ultralytics import YOLO

# Load model gốc (ví dụ đuôi .pt)
model = YOLO(".\\weights\\yolo11n-pose.pt") 

# Export với các tham số tương thích tuyệt đối với OpenCV
model.export(
    format="onnx", 
    opset=12,           # Ép dùng bộ phép toán cũ, ổn định cho OpenCV
    simplify=True,      # Gộp các phép toán thừa, cố định kích thước input
    dynamic=False       # Tắt tính năng kích thước ảnh tự do
)