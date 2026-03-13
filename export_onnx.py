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

# Kiểm tra file ONNX đã xuất có tương thích với Operators hay không
import onnx
model = onnx.load("weights/yolo11n-pose.onnx")   # hoặc đường dẫn file .onnx bạn có
onnx.checker.check_model(model)
for op in model.opset_import:
    print("domain:", op.domain, "version:", op.version)
    
# Kiểm tra chạy thử (small inference) ONNX với onnxruntime
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("weights/yolo11n-pose.onnx")
inp_name = sess.get_inputs()[0].name
# chuẩn bị mảng input phù hợp (ví dụ uint8 hoặc float32, shape phù hợp)
x = np.random.randn(1,3,640,640).astype(np.float32)
outs = sess.run(None, {inp_name: x})
print([o.shape for o in outs])