from ultralytics import YOLO

# 1. Load model PyTorch gốc
model = YOLO('.\\weights\\yolo11n.pt')

# 2. Export sang ONNX
# format='onnx': Định dạng xuất
# half=True: Sử dụng FP16 (giúp tăng FPS đáng kể trên GPU)
# dynamic=True: Cho phép kích thước ảnh đầu vào linh hoạt (nếu cần)
# imgsz=640: Cố định kích thước ảnh đầu vào (tối ưu tốc độ hơn dynamic)
model.export(format='onnx', half=True, imgsz=640)