import cv2
import numpy as np
import math
import csv
import os
import json
import time
from ultralytics import YOLO
import torch

# IMPORT MODULE CŨ
from height_estimator import HeightEstimator

# ================= CẤU HÌNH =================
VIDEO_PATH = '..\\test_imgs\\cam_2\\cam_2.mp4'  # <--- ĐỔI ĐƯỜNG DẪN VIDEO CỦA BẠN Ở ĐÂY
CALIB_FILE = '..\\calibration.json'
CONFIG_FILE = 'config.json'
CSV_FILE_NAME = 'video_measurement_data.csv'
TARGET_W = 1200  # Resize video về width này (phải khớp với logic lúc config)

class VideoDistanceApp:
    def __init__(self):
        self.mode = "DISTANCE" # DISTANCE hoặc HEIGHT
        self.paused = False    # Trạng thái tạm dừng
        
        # Tools
        self.height_tool = HeightEstimator()
        self.yolo_model = None
        
        # Data
        self.real_world = {}
        self.cam_real_pos = (0.5, -18.0)
        self.clicked_points = []     
        self.matrix_homography = None
        self.scale_px_per_meter = 1.0 
        
        # Calibration Maps (Tính 1 lần dùng mãi mãi)
        self.map1, self.map2 = None, None
        
        # Runtime variables
        self.current_frame = None
        self.detected_objects = []
        
        # Mouse interaction (cho Manual Measure)
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.measure_points = []
        self.height_clicks = []

        # Setup Device
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 0
            print(f"[INFO] GPU Activated: {torch.cuda.get_device_name(0)}")
        
        # Load Model
        try:
            self.yolo_model = YOLO('..\\weights\\yolo11n-pose.onnx') 
            print("[INFO] YOLO Model loaded.")
        except Exception as e:
            print(f"[ERR] Load YOLO failed: {e}")

    def init_calibration_maps(self, frame_size):
        """
        Tính toán Undistort Map 1 lần duy nhất để tối ưu FPS
        """
        if not os.path.exists(CALIB_FILE):
            print("[WARN] Không thấy file calibration. Chạy mode không undistort.")
            return

        try:
            with open(CALIB_FILE, 'r') as f: data = json.load(f)
            K = np.array(data['camera_matrix'])
            D = np.array(data['distortion_coefficients'])
            
            # Kích thước frame gốc từ video
            w_curr, h_curr = frame_size
            
            # Nếu file calib có resolution khác, cần scale K
            if 'image_resolution' in data:
                calib_w, calib_h = data['image_resolution']
                if w_curr != calib_w or h_curr != calib_h:
                    scale_x = w_curr / calib_w
                    scale_y = h_curr / calib_h
                    K[0, 0] *= scale_x; K[1, 1] *= scale_y
                    K[0, 2] *= scale_x; K[1, 2] *= scale_y

            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w_curr, h_curr), 1, (w_curr, h_curr))
            self.map1, self.map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w_curr, h_curr), 5)
            print("[INFO] Đã khởi tạo Calibration Maps.")
            
            # Load Focal Length cho Height Estimator luôn
            # Lưu ý: Pass vào TARGET_W vì sau này ta sẽ resize ảnh về size đó để xử lý
            self.height_tool.load_focal_length(CALIB_FILE, TARGET_W)
            
        except Exception as e:
            print(f"[ERR] Lỗi Calibration Init: {e}")

    def load_config(self):
        """Load homography và vùng ROI từ file config"""
        if not os.path.exists(CONFIG_FILE): return False
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                self.real_world = data['real_world']
                self.cam_real_pos = (data['camera']['real_x'], data['camera']['real_y'])
                self.scale_px_per_meter = data['settings'].get('scale_px_per_meter', 1.0)
                
                if 'points_px' in data:
                    self.clicked_points = [tuple(p) for p in data['points_px']]
                    self.compute_homography() # Tính matrix ngay
                    return True
        except Exception as e:
            print(f"[ERR] Load Config Failed: {e}")
        return False

    def get_quadrilateral_coords(self, l1, l2, l3, l4, d13):
        # (Copy hàm tính toán hình học từ main.py cũ)
        if l1 + l2 < d13 or abs(l1 - l2) > d13: return []
        p1 = (0.0, 0.0); p2 = (l1, 0.0)
        cos_alpha = (l1**2 + d13**2 - l2**2) / (2 * l1 * d13)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        alpha = math.acos(cos_alpha)
        p3 = (d13 * math.cos(alpha), d13 * math.sin(alpha))
        d = d13; a = (l4**2 - l3**2 + d**2) / (2*d)
        h = math.sqrt(max(0, l4**2 - a**2))
        x0 = p1[0] + a * (p3[0] - p1[0]) / d; y0 = p1[1] + a * (p3[1] - p1[1]) / d
        rx = -(p3[1] - p1[1]) / d; ry = (p3[0] - p1[0]) / d
        p4 = (x0 + h * rx, y0 + h * ry) # Simplified selection
        return [p1, p2, p3, p4]

    def compute_homography(self):
        if len(self.clicked_points) < 4: return
        rw = self.real_world
        real_coords = self.get_quadrilateral_coords(rw['L1'], rw['L2'], rw['L3'], rw['L4'], rw['diag_13'])
        if not real_coords: return
        
        dst_pts = np.float32([[pt[0]*self.scale_px_per_meter, pt[1]*self.scale_px_per_meter] for pt in real_coords])
        src_pts = np.float32(self.clicked_points)
        self.matrix_homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
        print("[INFO] Homography Matrix Updated.")

    def process_frame(self, raw_frame):
        # 1. Undistort
        if self.map1 is not None:
            frame = cv2.remap(raw_frame, self.map1, self.map2, cv2.INTER_LINEAR)
        else:
            frame = raw_frame

        # 2. Resize về kích thước làm việc (quan trọng để khớp với point trong config)
        h, w = frame.shape[:2]
        scale = TARGET_W / w
        new_h = int(h * scale)
        frame_resized = cv2.resize(frame, (TARGET_W, new_h))
        
        # 3. Detect Object
        self.detect_objects(frame_resized)
        
        return frame_resized

    def detect_objects(self, img):
        if self.yolo_model is None: return
        
        results = self.yolo_model(img, verbose=False, device=self.device, conf=0.5)
        self.detected_objects = []
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            kpts_data = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                
                # Logic lấy điểm đầu và chân (giống main.py)
                ground_point = (int((x1+x2)/2), y2)
                head_point = (int((x1+x2)/2), y1)
                
                if kpts_data is not None and len(kpts_data) > i:
                    kp = kpts_data[i]
                    # Chân (Ankles 15, 16)
                    if kp[15][2] > 0.5 and kp[16][2] > 0.5:
                        ground_point = (int((kp[15][0]+kp[16][0])/2), int((kp[15][1]+kp[16][1])/2))
                    # Mũi (Nose 0)
                    if kp[0][2] > 0.5:
                        head_point = (int(kp[0][0]), int(kp[0][1]))

                # Tự động tính chiều cao / khoảng cách ngay lập tức
                h_real, d_real = self.height_tool.calculate(head_point, ground_point, self.matrix_homography, self.cam_real_pos)

                self.detected_objects.append({
                    'box': box,
                    'head': head_point,
                    'foot': ground_point,
                    'h_real': h_real,
                    'd_real': d_real
                })

    def draw_overlays(self, img):
        # Vẽ vùng ROI đã config
        if len(self.clicked_points) == 4:
            pts = np.array(self.clicked_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 0, 255), 1, cv2.LINE_AA)

        # Vẽ detected objects
        for obj in self.detected_objects:
            x1, y1, x2, y2 = obj['box']
            head = obj['head']
            foot = obj['foot']
            
            # Box
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
            
            # Line Height
            cv2.line(img, head, foot, (0, 255, 0), 2)
            cv2.circle(img, head, 3, (0, 0, 255), -1)
            cv2.circle(img, foot, 3, (0, 255, 255), -1)
            
            # Text Info
            label = f"H:{obj['h_real']:.2f}m | D:{obj['d_real']:.1f}m"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Vẽ thông tin hệ thống
        status = "PAUSED (Space to Play)" if self.paused else "PLAYING (Space to Pause)"
        cv2.putText(img, f"MODE: {self.mode} | {status}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        # Vẽ manual measurement lines (nếu đang đo tay)
        if len(self.measure_points) > 0:
            for pt in self.measure_points:
                cv2.circle(img, pt, 4, (255, 0, 255), -1)
            if len(self.measure_points) >= 2:
                cv2.line(img, self.measure_points[-2], self.measure_points[-1], (0, 165, 255), 2)

# Global Instance để dùng cho Mouse Callback
app = VideoDistanceApp()

def mouse_event_video(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if app.mode == "DISTANCE":
            app.measure_points.append((x, y))
            # Nếu đủ 2 điểm -> Tính khoảng cách
            if len(app.measure_points) >= 2 and len(app.measure_points) % 2 == 0:
                p1, p2 = app.measure_points[-2], app.measure_points[-1]
                # Tính khoảng cách thực (dùng lại hàm tính toán nếu cần, hoặc tính trực tiếp ở đây)
                if app.matrix_homography is not None:
                     pts = np.float32([p1, p2]).reshape(-1, 1, 2)
                     trans_pts = cv2.perspectiveTransform(pts, app.matrix_homography)
                     dist_px = np.linalg.norm(trans_pts[0][0] - trans_pts[1][0])
                     dist_m = dist_px / app.scale_px_per_meter
                     print(f"[MANUAL] Distance: {dist_m:.2f} m")

def main():
    # 1. Setup Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Không mở được video: {VIDEO_PATH}")
        return

    # 2. Setup Calibration & Config
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    app.init_calibration_maps((width, height))
    
    if not app.load_config():
        print("[WARN] Chưa load được config.json. Số liệu sẽ sai.")

    cv2.namedWindow("Video Analysis")
    cv2.setMouseCallback("Video Analysis", mouse_event_video)

    print("\n--- HƯỚNG DẪN VIDEO ---")
    print(" - [Space]: Tạm dừng / Tiếp tục")
    print(" - [d]: Chế độ đo khoảng cách (Click 2 điểm khi Pause)")
    print(" - [q]: Thoát")

    while True:
        if not app.paused:
            ret, frame = cap.read()
            if not ret:
                print("Hết video.")
                break # Hoặc cap.set(cv2.CAP_PROP_POS_FRAMES, 0) để loop
            
            # Xử lý frame (Undistort -> Resize -> Detect)
            app.current_frame = app.process_frame(frame)
        
        # Luôn vẽ lại overlay trên frame hiện tại (kể cả khi pause)
        display_img = app.current_frame.copy()
        app.draw_overlays(display_img)
        
        cv2.imshow("Video Analysis", display_img)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        if key == ord(' '): # Phím cách để Pause/Play
            app.paused = not app.paused
        if key == ord('d'): app.mode = "DISTANCE"
        if key == ord('r'): app.measure_points = [] # Reset điểm đo tay

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()