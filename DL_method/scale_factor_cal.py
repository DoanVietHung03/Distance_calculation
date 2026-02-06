import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

# ================= CẤU HÌNH NGƯỜI DÙNG =================
# FILE ẢNH CẦN TEST
IMAGE_SOURCE = "..\\test_imgs\\cam_2\\cam_2_near.jpg"  # Đường dẫn file ảnh

# NHẬP KHOẢNG CÁCH THỰC TẾ BẠN ĐÃ ĐO ĐƯỢC VÀO ĐÂY (Mét)
# Ví dụ: Từ camera đến chỗ người đứng là 18.5 mét
REAL_DISTANCE_METERS = 6.0 

# Cấu hình Model
# Lưu ý: Với người ở xa, model 'Base' sẽ chính xác hơn 'Small'
DEPTH_MODEL_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
YOLO_MODEL_PATH = "..\\weights\\yolo11n.onnx" 

TARGET_HEIGHT = 640 # Resize ảnh đầu vào để xử lý

# ================= CLASS XỬ LÝ ĐỘ SÂU (DEPTH AI) =================
class DepthEstimator:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.device_name = "GPU (CUDA)" if self.device == 0 else "CPU"
        print(f"[INFO] Đang load Depth Anything V2 trên: {self.device_name}...")
        
        # Load Pipeline
        self.pipe = pipeline(task="depth-estimation", 
                             model=DEPTH_MODEL_REPO, 
                             device=self.device)
        print("[INFO] Load Depth Model xong!")

    def predict(self, frame_cv2):
        """Input: Ảnh BGR -> Output: Depth Map (Numpy)"""
        image_pil = Image.fromarray(cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2RGB))
        
        depth_output = self.pipe(image_pil)
        depth_tensor = depth_output["predicted_depth"]
        
        depth_map = depth_tensor.squeeze().cpu().numpy()
        h, w = frame_cv2.shape[:2]
        
        # Resize depth map về đúng kích thước ảnh gốc
        depth_map_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        return depth_map_resized

# ================= CLASS CHÍNH (APP) =================
class SmartDistanceApp:
    def __init__(self):
        # 1. Đọc ảnh thay vì VideoCapture
        self.frame = cv2.imread(IMAGE_SOURCE)
        if self.frame is None:
            print(f"[ERR] Không tìm thấy file ảnh: {IMAGE_SOURCE}")
            exit()
            
        # 2. Khởi tạo Models
        self.depth_engine = DepthEstimator()
        
        print(f"[INFO] Đang load YOLO ({YOLO_MODEL_PATH})...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)

    def run(self):
        print("\n--- CHẾ ĐỘ CALIBRATION (ẢNH TĨNH) ---")
        
        # 1. Resize ảnh để hiển thị và xử lý cho nhanh (nếu ảnh quá to)
        h, w = self.frame.shape[:2]
        # Giữ nguyên độ phân giải gốc để đo cho chính xác, chỉ resize khi hiển thị sau

        # 2. CHẠY DEPTH AI
        print("[1/3] Đang tính bản đồ độ sâu (Depth Map)...")
        depth_map = self.depth_engine.predict(self.frame)

        # 3. CHẠY YOLO
        print("[2/3] Đang phát hiện người...")
        results = self.yolo_model(self.frame, classes=[0], verbose=False) # Class 0 = Person

        # 4. TÍNH TOÁN VÀ HIỂN THỊ
        print("[3/3] Đang tính toán Scale Factor...")
        
        found_person = False
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Lấy tọa độ Box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # --- CHIẾN THUẬT LẤY ĐIỂM ĐO ---
                # Với ảnh xa, lấy tâm box (center) thường ổn định hơn chân
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Lấy giá trị Raw Depth tại tâm
                raw_val = depth_map[cy, cx]
                
                # --- TÍNH TOÁN CALIBRATION ---
                # Công thức: Scale = Khoảng_Cách_Thực * Giá_Trị_Raw
                suggested_scale = REAL_DISTANCE_METERS * raw_val
                
                # In ra Terminal
                print("\n" + "="*40)
                print(f" KẾT QUẢ TẠI NGƯỜI VỊ TRÍ ({cx}, {cy})")
                print(f" - Khoảng cách thực nhập vào : {REAL_DISTANCE_METERS} mét")
                print(f" - Giá trị Raw AI đo được    : {raw_val:.4f}")
                print("-" * 40)
                print(f" >>> SCALE_FACTOR GỢI Ý      : {suggested_scale:.2f}")
                print("="*40 + "\n")
                
                # Vẽ lên ảnh
                cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Ghi thông số lên ảnh
                text_info = [
                    f"Raw Val: {raw_val:.1f}",
                    f"Real Dist: {REAL_DISTANCE_METERS}m",
                    f"Scale Factor: {suggested_scale:.1f}"
                ]
                
                for i, line in enumerate(text_info):
                    cv2.putText(self.frame, line, (x1, y1 - 10 - (i*25)), font, 0.6, (0, 255, 255), 2)
                
                cv2.circle(self.frame, (cx, cy), 5, (0, 0, 255), -1)
                found_person = True

        if not found_person:
            print("[WARN] Không tìm thấy người nào trong ảnh!")
        else:
            print("Đã xong! Hãy lấy số 'SCALE_FACTOR GỢI Ý' ở trên để điền vào code Camera.")

        # Hiển thị Depth Map (để debug xem có bị nhiễu không)
        depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
        
        # Resize để hiển thị vừa màn hình laptop
        display_h = 700
        scale_disp = display_h / h
        disp_w = int(w * scale_disp)
        
        img_show = cv2.resize(self.frame, (disp_w, display_h))
        depth_show = cv2.resize(depth_vis, (disp_w, display_h))
        
        # Ghép ảnh lại xem cho sướng
        combined = np.hstack((img_show, depth_show))
        
        cv2.imshow("Calibration Result (Press Any Key to Exit)", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SmartDistanceApp()
    app.run()