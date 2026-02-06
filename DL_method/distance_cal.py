import cv2
import torch
import numpy as np
import math
from transformers import pipeline
from PIL import Image

# ================= CẤU HÌNH NGƯỜI DÙNG =================
IMAGE_PATH = "..\\test_imgs\\cam_2\\cam_2_near.jpg" 

# 1. HỆ SỐ SCALE (Lấy từ bước Calibrate trước)
SCALE_FACTOR = 12.82  # <--- THAY SỐ CỦA BẠN VÀO ĐÂY

# 2. CẤU HÌNH HIỂN THỊ (RESIZE)
# Thu nhỏ cửa sổ lại cho dễ nhìn (ví dụ chiều ngang 1000 pixel)
DISPLAY_WIDTH = 1000 

# 3. THÔNG SỐ CAMERA (PINHOLE)
# Nếu bạn biết tiêu cự thật (từ calibration.json) thì điền vào đây.
# Nếu không biết: Để None, code sẽ tự ước lượng (Focal Length ~= Width của ảnh)
FOCAL_LENGTH = 1600 

# Model AI
DEPTH_MODEL_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
# ========================================================

class Measure3DTool:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"[Init] Đang load model trên {'GPU' if self.device==0 else 'CPU'}...")
        
        # Load Model
        self.pipe = pipeline(task="depth-estimation", model=DEPTH_MODEL_REPO, device=self.device)
        
        # Load Ảnh gốc
        self.original_image = cv2.imread(IMAGE_PATH)
        if self.original_image is None:
            print(f"[ERR] Không tìm thấy ảnh: {IMAGE_PATH}")
            exit()
            
        # --- XỬ LÝ RESIZE (Để hiển thị không bị tràn màn hình) ---
        h_org, w_org = self.original_image.shape[:2]
        scale_ratio = DISPLAY_WIDTH / w_org
        self.new_h = int(h_org * scale_ratio)
        self.new_w = DISPLAY_WIDTH
        
        print(f"[Resize] Ảnh gốc: {w_org}x{h_org} -> Hiển thị: {self.new_w}x{self.new_h}")
        self.image = cv2.resize(self.original_image, (self.new_w, self.new_h))
        
        # Cập nhật thông số Pinhole theo ảnh đã resize
        self.cx = self.new_w / 2
        self.cy = self.new_h / 2
        
        # Ước lượng tiêu cự nếu người dùng không nhập
        # (Theo kinh nghiệm: Focal length xấp xỉ chiều ngang ảnh với camera góc nhìn ~50-60 độ)
        if FOCAL_LENGTH is None:
            self.fx = self.new_w * 0.9  # Ước lượng 0.9 - 1.0 lần chiều rộng
            self.fy = self.new_w * 0.9
        else:
            # Scale tiêu cự theo tỉ lệ resize ảnh
            self.fx = FOCAL_LENGTH * scale_ratio
            self.fy = FOCAL_LENGTH * scale_ratio
            
        self.depth_map = None
        self.points = [] # Lưu danh sách điểm click [(x,y,z), (x,y,z)]

    def process_depth(self):
        print("[Processing] Đang tính toán Depth Map...")
        # Chuyển sang PIL
        img_pil = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        
        # Inference
        depth_output = self.pipe(img_pil)
        depth_tensor = depth_output["predicted_depth"]
        depth_map_raw = depth_tensor.squeeze().cpu().numpy()
        
        # Resize Depth Map khớp với ảnh hiển thị
        self.depth_map = cv2.resize(depth_map_raw, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)
        print("[Done] Đã xong! Hãy click 2 điểm để đo khoảng cách.")

    def pixel_to_3d(self, u, v, raw_depth_val):
        """Chuyển đổi Pixel (u,v) + Depth -> Tọa độ 3D (X, Y, Z)"""
        if raw_depth_val <= 0: return None
        
        # 1. Tính Z (Mét)
        Z = SCALE_FACTOR / raw_depth_val
        
        # 2. Tính X, Y (Mét) dùng công thức Pinhole
        # X = (u - cx) * Z / fx
        X = (u - self.cx) * Z / self.fx
        # Y = (v - cy) * Z / fy
        Y = (v - self.cy) * Z / self.fy
        
        return np.array([X, Y, Z])

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Lấy giá trị depth
            raw_val = self.depth_map[y, x]
            coord_3d = self.pixel_to_3d(x, y, raw_val)
            
            if coord_3d is not None:
                self.points.append((x, y, coord_3d))
                print(f"Click [{len(self.points)}]: Pixel({x},{y}) -> 3D({coord_3d[0]:.2f}, {coord_3d[1]:.2f}, {coord_3d[2]:.2f})")
                
                # Vẽ lại giao diện
                self.draw_ui()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Chuột phải để Reset
            self.points = []
            print("[Reset] Đã xóa các điểm đo.")
            self.draw_ui()

    def draw_ui(self):
        img_display = self.image.copy()
        
        # Vẽ các điểm đã click
        for i, (px, py, c3d) in enumerate(self.points):
            color = (0, 0, 255) if i == 0 else (255, 0, 0) # Điểm 1 Đỏ, Điểm 2 Xanh
            cv2.circle(img_display, (px, py), 5, color, -1)
            cv2.putText(img_display, f"P{i+1}", (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Nếu đã có đủ 2 điểm -> Tính khoảng cách
        if len(self.points) >= 2:
            p1 = self.points[-2] # Điểm áp chót
            p2 = self.points[-1] # Điểm cuối cùng
            
            # Tọa độ 3D
            xyz1 = p1[2]
            xyz2 = p2[2]
            
            # Tính khoảng cách Euclidean trong không gian 3D
            dist_3d = np.linalg.norm(xyz1 - xyz2)
            
            # Vẽ đường nối
            cv2.line(img_display, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 255), 2)
            
            # Hiển thị số đo ở trung điểm đoạn thẳng
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            
            label = f"{dist_3d:.2f}m"
            # Vẽ nền đen cho chữ dễ đọc
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img_display, (mid_x, mid_y - h - 5), (mid_x + w, mid_y + 5), (0, 0, 0), -1)
            cv2.putText(img_display, label, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # In chi tiết ra console
            print(f"--> Khoảng cách P{len(self.points)-1}-P{len(self.points)}: {dist_3d:.2f} mét")

        cv2.imshow("Measure 3D Distance (Left: Click, Right: Reset)", img_display)

    def run(self):
        self.process_depth()
        
        cv2.namedWindow("Measure 3D Distance (Left: Click, Right: Reset)")
        cv2.setMouseCallback("Measure 3D Distance (Left: Click, Right: Reset)", self.mouse_callback)
        
        self.draw_ui() # Vẽ lần đầu
        
        print("\n--- HƯỚNG DẪN ---")
        print("1. Chuột TRÁI: Chọn điểm (A, B, C...)")
        print("2. Chuột PHẢI: Xóa hết làm lại")
        print("3. Nhấn 'q': Thoát")
        
        while True:
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tool = Measure3DTool()
    tool.run()