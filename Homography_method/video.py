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
VIDEO_PATH = '..\\test_imgs\\cam_2\\cam_2.mp4'  # <--- ĐỔI ĐƯỜNG DẪN VIDEO CỦA BẠN
CALIB_FILE = '..\\calibration.json'
CONFIG_FILE = 'config.json'
TARGET_W = 1200  # Resize video về width này

class VideoDistanceApp:
    def __init__(self):
        # --- CẤU HÌNH LOGIC MỚI ---
        self.mode = "DISTANCE" # Mặc định là đo khoảng cách tới điểm click
        self.paused = False    # Trạng thái tạm dừng
        
        self.target_point = None # Điểm đích (Target) do người dùng click
        
        # Tools
        self.height_tool = HeightEstimator()
        self.yolo_model = None
        
        # Data
        self.real_world = {}
        self.cam_real_pos = (0.5, -18.0)
        self.clicked_points = []     
        self.matrix_homography = None
        self.scale_px_per_meter = 1.0 
        
        # Calibration Maps
        self.map1, self.map2 = None, None
        
        # Runtime variables
        self.current_frame = None
        self.detected_objects = [] # Danh sách người detect được
        
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
        if not os.path.exists(CALIB_FILE):
            return
        try:
            with open(CALIB_FILE, 'r') as f: data = json.load(f)
            K = np.array(data['camera_matrix'])
            D = np.array(data['distortion_coefficients'])
            w_curr, h_curr = frame_size
            if 'image_resolution' in data:
                calib_w, calib_h = data['image_resolution']
                if w_curr != calib_w or h_curr != calib_h:
                    scale_x = w_curr / calib_w
                    scale_y = h_curr / calib_h
                    K[0, 0] *= scale_x; K[1, 1] *= scale_y; K[0, 2] *= scale_x; K[1, 2] *= scale_y
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w_curr, h_curr), 1, (w_curr, h_curr))
            self.map1, self.map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w_curr, h_curr), 5)
            self.height_tool.load_focal_length(CALIB_FILE, TARGET_W)
        except Exception as e:
            print(f"[ERR] Lỗi Calibration Init: {e}")

    def load_config(self):
        if not os.path.exists(CONFIG_FILE): return False
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                self.real_world = data['real_world']
                self.cam_real_pos = (data['camera']['real_x'], data['camera']['real_y'])
                self.scale_px_per_meter = data['settings'].get('scale_px_per_meter', 1.0)
                if 'points_px' in data:
                    self.clicked_points = [tuple(p) for p in data['points_px']]
                    self.compute_homography()
                    return True
        except Exception as e:
            print(f"[ERR] Load Config Failed: {e}")
        return False

    def get_quadrilateral_coords(self, l1, l2, l3, l4, d13):
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
        p4 = (x0 + h * rx, y0 + h * ry)
        return [p1, p2, p3, p4]

    def compute_homography(self):
        if len(self.clicked_points) < 4: return
        rw = self.real_world
        real_coords = self.get_quadrilateral_coords(rw['L1'], rw['L2'], rw['L3'], rw['L4'], rw['diag_13'])
        if not real_coords: return
        dst_pts = np.float32([[pt[0]*self.scale_px_per_meter, pt[1]*self.scale_px_per_meter] for pt in real_coords])
        src_pts = np.float32(self.clicked_points)
        self.matrix_homography = cv2.getPerspectiveTransform(src_pts, dst_pts)

    def calculate_distance_points(self, p1, p2):
        """Tính khoảng cách thực giữa 2 điểm trên ảnh"""
        if self.matrix_homography is None: return 0.0
        pts = np.float32([p1, p2]).reshape(-1, 1, 2)
        trans_pts = cv2.perspectiveTransform(pts, self.matrix_homography)
        dist_px = np.linalg.norm(trans_pts[0][0] - trans_pts[1][0])
        return dist_px / self.scale_px_per_meter

    def process_frame(self, raw_frame):
        # Undistort & Resize
        if self.map1 is not None:
            frame = cv2.remap(raw_frame, self.map1, self.map2, cv2.INTER_LINEAR)
        else: frame = raw_frame
        h, w = frame.shape[:2]
        scale = TARGET_W / w
        new_h = int(h * scale)
        frame_resized = cv2.resize(frame, (TARGET_W, new_h))
        
        # Detect Object
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
                ground_point = (int((x1+x2)/2), y2) # Mặc định chân là giữa đáy box
                head_point = (int((x1+x2)/2), y1)   # Mặc định đầu là giữa đỉnh box
                
                if kpts_data is not None and len(kpts_data) > i:
                    kp = kpts_data[i]
                    # Nếu có keypoint chân -> chính xác hơn
                    if kp[15][2] > 0.5 and kp[16][2] > 0.5:
                        ground_point = (int((kp[15][0]+kp[16][0])/2), int((kp[15][1]+kp[16][1])/2))
                    if kp[0][2] > 0.5:
                        head_point = (int(kp[0][0]), int(kp[0][1]))

                # Lưu thông tin object
                obj_info = {
                    'box': box,
                    'head': head_point,
                    'foot': ground_point,
                    'h_real': 0.0,
                    'd_to_target': 0.0
                }
                
                # --- TÍNH TOÁN TÙY THEO CHẾ ĐỘ ---
                if self.mode == "HEIGHT":
                    # Chỉ tính chiều cao khi ở mode Height
                    h_real, _ = self.height_tool.calculate(head_point, ground_point, self.matrix_homography, self.cam_real_pos)
                    obj_info['h_real'] = h_real
                
                elif self.mode == "DISTANCE":
                    # Chỉ tính khoảng cách nếu đã có Target Point
                    if self.target_point is not None:
                        d_target = self.calculate_distance_points(ground_point, self.target_point)
                        obj_info['d_to_target'] = d_target
                
                self.detected_objects.append(obj_info)

    def draw_overlays(self, img):
        # 1. Vẽ vùng đo (Reference)
        if len(self.clicked_points) == 4:
            pts = np.array(self.clicked_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 0, 100), 1, cv2.LINE_AA)

        # 2. Vẽ Target Point (nếu có)
        if self.mode == "DISTANCE" and self.target_point:
            cv2.circle(img, self.target_point, 6, (0, 255, 255), -1)
            cv2.circle(img, self.target_point, 10, (0, 255, 255), 2)
            cv2.putText(img, "TARGET", (self.target_point[0]+10, self.target_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 3. Vẽ thông tin Người
        for obj in self.detected_objects:
            x1, y1, x2, y2 = obj['box']
            head = obj['head']
            foot = obj['foot']
            
            # Vẽ Box mờ
            cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), 1)

            if self.mode == "HEIGHT":
                # HIỂN THỊ CHIỀU CAO
                cv2.line(img, head, foot, (0, 255, 0), 2)
                cv2.putText(img, f"H: {obj['h_real']:.2f}m", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            elif self.mode == "DISTANCE":
                # HIỂN THỊ KHOẢNG CÁCH TỚI TARGET
                cv2.circle(img, foot, 4, (255, 0, 0), -1) # Điểm chân
                if self.target_point:
                    # Vẽ đường nối từ chân người tới Target
                    cv2.line(img, foot, self.target_point, (0, 165, 255), 2)
                    
                    # Tính trung điểm để hiển thị text
                    mid_x = (foot[0] + self.target_point[0]) // 2
                    mid_y = (foot[1] + self.target_point[1]) // 2
                    
                    cv2.putText(img, f"{obj['d_to_target']:.2f}m", (mid_x, mid_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, "Click to set Target", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # 4. Vẽ UI Status
        status_text = "PAUSED (Switch Mode Allowed)" if self.paused else "PLAYING (Space to Pause)"
        color_status = (0, 0, 255) if self.paused else (0, 255, 0)
        
        cv2.rectangle(img, (0, 0), (TARGET_W, 60), (0, 0, 0), -1)
        cv2.putText(img, f"MODE: {self.mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(img, status_text, (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)

app = VideoDistanceApp()

def mouse_event_video(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Click chuột để đặt Target Point cho chế độ Distance
        if app.mode == "DISTANCE":
            app.target_point = (x, y)
            print(f"[TARGET] Đã đặt điểm đích tại: {app.target_point}")

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Không mở được video.")
        return

    # Init
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    app.init_calibration_maps((width, height))
    app.load_config()

    cv2.namedWindow("Smart Analysis")
    cv2.setMouseCallback("Smart Analysis", mouse_event_video)

    print("\n--- HƯỚNG DẪN ---")
    print("1. Mặc định: Click chuột lên sàn để đo khoảng cách từ người tới điểm đó.")
    print("2. Nhấn SPACE để Pause.")
    print("3. Khi Pause, nhấn 'h' để xem Chiều cao, 'd' để về đo Khoảng cách.")

    while True:
        if not app.paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                continue
            
            app.current_frame = app.process_frame(frame)
        
        # Luôn vẽ Overlay (kể cả khi Pause)
        display_img = app.current_frame.copy()
        app.draw_overlays(display_img)
        cv2.imshow("Smart Analysis", display_img)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        
        # --- LOGIC CHUYỂN MODE (CHỈ KHI PAUSE) ---
        if key == ord(' '): 
            app.paused = not app.paused
            
        if app.paused:
            if key == ord('h'): 
                app.mode = "HEIGHT"
                print("-> Switched to HEIGHT mode.")
            if key == ord('d'): 
                app.mode = "DISTANCE"
                print("-> Switched to DISTANCE mode.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()