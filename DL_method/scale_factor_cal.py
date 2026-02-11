import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

# ================= CẤU HÌNH NGƯỜI DÙNG =================
IMAGE_SOURCE = "..\\test_imgs\\cam_2\\frame_test.jpg"
REAL_DISTANCE_METERS = 15.3 

DEPTH_MODEL_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
YOLO_MODEL_PATH = "..\\weights\\yolo11n.onnx" 

# QUAN TRỌNG: Phải khớp với cấu hình trong distance_cal.py
PROCESS_WIDTH = 640  
# ========================================================

class DepthEstimator:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(task="depth-estimation", model=DEPTH_MODEL_REPO, device=self.device)

    def predict(self, frame_cv2):
        h_org, w_org = frame_cv2.shape[:2]
        
        # 1. Resize input cho giống hệt lúc chạy Video Realtime
        # Tính chiều cao mới giữ nguyên tỷ lệ khung hình
        new_h = int(h_org * (PROCESS_WIDTH / w_org))
        frame_resized = cv2.resize(frame_cv2, (PROCESS_WIDTH, new_h))
        
        # 2. Đưa vào AI xử lý
        image_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        depth_output = self.pipe(image_pil)
        depth_tensor = depth_output["predicted_depth"]
        depth_map_small = depth_tensor.squeeze().cpu().numpy()
        
        # 3. Resize kết quả depth map về lại kích thước gốc để khớp với tọa độ YOLO
        depth_map_final = cv2.resize(depth_map_small, (w_org, h_org), interpolation=cv2.INTER_LINEAR)
        
        return depth_map_final

class SmartDistanceApp:
    def __init__(self):
        self.frame = cv2.imread(IMAGE_SOURCE)
        if self.frame is None:
            print(f"[ERR] Không tìm thấy file: {IMAGE_SOURCE}")
            exit()
        self.depth_engine = DepthEstimator()
        self.yolo_model = YOLO(YOLO_MODEL_PATH)

    def run(self):
        print(f"--- ĐANG CALIBRATE VỚI INPUT WIDTH = {PROCESS_WIDTH} ---")
        
        # Bước 1: Tính Depth (đã bao gồm resize bên trong hàm predict để đồng bộ)
        depth_map = self.depth_engine.predict(self.frame)

        # Bước 2: Chạy YOLO trên ảnh gốc (YOLO tự resize bên trong nên không lo)
        results = self.yolo_model(self.frame, classes=[0], verbose=False)

        # Bước 3: Tính toán
        found = False
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Lấy giá trị raw từ depth map
                raw_val = depth_map[cy, cx]
                
                scale_factor = REAL_DISTANCE_METERS * raw_val
                
                print(f"\n>> NGƯỜI TẠI ({cx}, {cy})")
                print(f"   Raw Depth (tại {PROCESS_WIDTH}px input): {raw_val:.4f}")
                print(f"   Khoảng cách thực: {REAL_DISTANCE_METERS}m")
                print(f"   >>> SCALE FACTOR CẦN DÙNG: {scale_factor:.2f}")
                
                cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.frame, f"Scale: {scale_factor:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                found = True

        if found:
            # Resize hiển thị cho vừa màn hình (Cái này KHÔNG ảnh hưởng tính toán)
            disp_h = 700
            scale = disp_h / self.frame.shape[0]
            disp_w = int(self.frame.shape[1] * scale)
            cv2.imshow("Result", cv2.resize(self.frame, (disp_w, disp_h)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Không tìm thấy người!")

if __name__ == "__main__":
    SmartDistanceApp().run()