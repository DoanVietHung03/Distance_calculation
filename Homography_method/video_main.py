import cv2
import numpy as np
import math
import json
import time
import threading
import queue
import os
from ultralytics import YOLO
import torch

# --- IMPORT MODULES ---
# Giả sử bạn để file stabilizer.py cùng thư mục
try:
    from stabilizer import ROIStabilizer
except ImportError:
    print("[WARN] Không tìm thấy stabilizer.py, sẽ tắt tính năng chống rung.")
    ROIStabilizer = None

# ================= CẤU HÌNH =================
VIDEO_PATH = '..\\test_imgs\\cam_2\\test.mp4' 
CALIB_FILE = '..\\calibration.json'
CONFIG_FILE = 'config.json'
TARGET_W = 1200  # Resize về kích thước này để xử lý cho nhanh

# ================= CLASS XỬ LÝ AI (Update logic từ main.py) =================
class YOLOThread(threading.Thread):
    def __init__(self, model_path, device):
        threading.Thread.__init__(self)
        self.model_path = model_path
        self.device = device
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.stopped = False
        self.model = None

    def load_model(self):
        try:
            print("[AI] Loading YOLO model...")
            self.model = YOLO(self.model_path)
            # Warmup
            self.model(np.zeros((640,640,3), dtype=np.uint8), verbose=False, device=self.device)
            print("[AI] Model Ready.")
        except Exception as e:
            print(f"[AI ERR] Không load được model: {e}")

    def run(self):
        self.load_model()
        while not self.stopped:
            try:
                frame = self.input_queue.get(timeout=0.1)
                if self.model:
                    # Chạy inference
                    results = self.model(frame, verbose=False, device=self.device)
                    processed = self.process_results(results, frame.shape[1], frame.shape[0])
                    
                    if not self.output_queue.empty():
                        try: self.output_queue.get_nowait()
                        except queue.Empty: pass
                    self.output_queue.put(processed)
                self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AI LOOP ERR] {e}")

    def process_results(self, results, img_width, img_height):
        detected_objects = []
        
        # [NEW] Margin để lọc lỗi viền đen do Undistort
        # Nếu box chạm vào vùng rìa phải (vùng đen), bỏ qua
        RIGHT_MARGIN = 80 
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            has_kpts = (r.keypoints is not None and r.keypoints.data is not None)
            keypoints = r.keypoints.data.cpu().numpy() if has_kpts else None

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                w_box = x2 - x1
                h_box = y2 - y1
                
                # 1. Lọc nhiễu kích thước
                if w_box < 20 or h_box < 20: continue
                
                # 2. [NEW] Lọc nhiễu viền đen (Fix lỗi box trắng bên phải)
                if x2 > (img_width - RIGHT_MARGIN): continue
                if x1 < 20: continue # Lọc viền trái nếu cần
                
                ground_point = None
                
                # 3. [NEW] Logic IS_FAR_AWAY giống main.py
                # Nếu người quá nhỏ (< 60px), coi là ở xa -> Dùng BBox cho ổn định
                IS_FAR_AWAY = h_box < 60
                
                if not IS_FAR_AWAY and keypoints is not None and len(keypoints) > i:
                    kpts = keypoints[i]
                    left_ankle = kpts[15]
                    right_ankle = kpts[16]
                    
                    l_conf = left_ankle[2]
                    r_conf = right_ankle[2]

                    if l_conf > 0.5 and r_conf > 0.5:
                        gx = int((left_ankle[0] + right_ankle[0]) / 2)
                        gy = int((left_ankle[1] + right_ankle[1]) / 2)
                        ground_point = (gx, gy)
                    elif l_conf > 0.5:
                        ground_point = (int(left_ankle[0]), int(left_ankle[1]))
                    elif r_conf > 0.5:
                        ground_point = (int(right_ankle[0]), int(right_ankle[1]))

                # Fallback: Dùng đáy Box
                if ground_point is None:
                    ground_point = (int((x1 + x2) / 2), int(y2))
                
                detected_objects.append({
                    'box': box,
                    'ground_point': ground_point,
                    'is_far': IS_FAR_AWAY
                })
        return detected_objects

    def stop(self):
        self.stopped = True

# ================= CLASS CHÍNH: VIDEO MEASUREMENT =================
class VideoDistanceApp:
    def __init__(self):
        # Thông số đo đạc
        self.matrix_homography = None
        self.scale_px_per_meter = 1.0
        self.real_world = {} 
        
        # State
        self.step = "SELECT_ROI"
        self.clicked_points = [] 
        self.target_point = None 
        self.original_target_point = None
        
        # Stabilizer
        self.stabilizer = ROIStabilizer() if ROIStabilizer else None
        self.use_stabilizer = True

        # Load Data
        self.calib_data = self.load_calibration_data()
        self.load_config(CONFIG_FILE)
        
        # Setup AI
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Running on: {torch.cuda.get_device_name(0) if self.device == 0 else 'CPU'}")
        
        self.ai_thread = YOLOThread('..\\weights\\yolo11n-pose.onnx', self.device)
        self.ai_thread.daemon = True
        self.ai_thread.start()
        
        self.latest_detections = []

    def load_calibration_data(self):
        if os.path.exists(CALIB_FILE):
            try:
                with open(CALIB_FILE, 'r') as f: return json.load(f)
            except: pass
        return None

    # [NEW] Logic Undistort thông minh từ main.py
    def smart_undistort(self, img):
        if self.calib_data is None: return img
        try:
            h_curr, w_curr = img.shape[:2]
            K = np.array(self.calib_data['camera_matrix'])
            D = np.array(self.calib_data['distortion_coefficients'])
            
            # Kiểm tra resolution gốc trong file json
            if 'image_resolution' in self.calib_data:
                calib_w, calib_h = self.calib_data['image_resolution']
            else:
                calib_w, calib_h = w_curr, h_curr # Fallback

            # Nếu kích thước video khác kích thước calibrate -> Scale ma trận K
            if w_curr != calib_w or h_curr != calib_h:
                scale_x = w_curr / calib_w
                scale_y = h_curr / calib_h
                K[0, 0] *= scale_x
                K[1, 1] *= scale_y
                K[0, 2] *= scale_x
                K[1, 2] *= scale_y
            
            # Tạo map undistort tối ưu
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w_curr, h_curr), 1, (w_curr, h_curr))
            
            # Dùng remap (nhanh hơn và chuẩn hơn undistort thường trong loop)
            map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w_curr, h_curr), 5)
            undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            return undistorted
        except Exception as e:
            print(f"[ERR] Undistort Error: {e}")
            return img

    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                self.real_world = data['real_world']
                self.scale_px_per_meter = data['settings'].get('scale_px_per_meter', 1.0)
                print(f"[CONFIG] Loaded Real World: {self.real_world}")
        except Exception as e: 
            print(f"[ERR] Config Load Error: {e}")

    # Logic tính toán tọa độ thực từ L1, L2... (Giữ nguyên vì đã đúng)
    def get_real_coords_from_params(self):
        rw = self.real_world
        l1, l2, l3, l4, d13 = rw.get('L1',0), rw.get('L2',0), rw.get('L3',0), rw.get('L4',0), rw.get('diag_13',0)
        if l1==0: return None

        p1 = (0.0, 0.0)
        p2 = (l1, 0.0)
        try:
            cos_a = (l1**2 + d13**2 - l2**2)/(2*l1*d13)
            cos_a = max(-1.0, min(1.0, cos_a))
            alpha = math.acos(cos_a)
            p3 = (d13 * math.cos(alpha), d13 * math.sin(alpha))
            
            d = d13
            a = (l4**2 - l3**2 + d**2) / (2*d)
            h_val = math.sqrt(max(0, l4**2 - a**2))
            
            x2, y2 = p3
            x0 = p1[0] + a * (x2 - p1[0]) / d
            y0 = p1[1] + a * (y2 - p1[1]) / d
            rx = -(y2 - p1[1]) / d
            ry = (x2 - p1[0]) / d
            
            p4_a = (x0 + h_val * rx, y0 + h_val * ry)
            p4_b = (x0 - h_val * rx, y0 - h_val * ry)
            
            def cross_product(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

            if cross_product(p1, p3, p4_a) < 0: p4 = p4_a 
            else: p4 = p4_b
            
            return [p1, p2, p3, p4]
        except: return None

    def update_homography(self, current_roi_points):
        real_coords = self.get_real_coords_from_params()
        if real_coords and len(current_roi_points) == 4:
            dst_pts = np.float32([[pt[0] * self.scale_px_per_meter, pt[1] * self.scale_px_per_meter] for pt in real_coords])
            src_pts = np.float32(current_roi_points)
            try:
                self.matrix_homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
                print("[INFO] Homography Updated.")
                return True
            except: return False
        return False

    def calculate_distance_real(self, p_target, p_foot):
        if self.matrix_homography is None: return 0.0
        pts = np.float32([p_target, p_foot]).reshape(-1, 1, 2)
        trans_pts = cv2.perspectiveTransform(pts, self.matrix_homography)
        dist_px = np.linalg.norm(trans_pts[0][0] - trans_pts[1][0])
        return dist_px / self.scale_px_per_meter

    def run_video(self):
        if not os.path.exists(VIDEO_PATH):
            print(f"[ERR] Không tìm thấy video: {VIDEO_PATH}")
            return

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened(): return

        # --- PREPARE FIRST FRAME ---
        ret, first_frame = cap.read()
        if not ret: return

        # 1. Undistort frame đầu tiên (Dùng smart_undistort mới)
        undist = self.smart_undistort(first_frame)
        
        # 2. Resize về TARGET_W (Ví dụ 1200) để đồng bộ với logic xử lý
        h_orig, w_orig = undist.shape[:2]
        scale_factor = TARGET_W / w_orig
        new_h = int(h_orig * scale_factor)
        
        first_frame_resized = cv2.resize(undist, (TARGET_W, new_h))
        current_frame = first_frame_resized.copy()
        
        # --- MOUSE CALLBACK ---
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.step == "SELECT_ROI":
                    if len(self.clicked_points) < 4:
                        self.clicked_points.append((x, y))
                        print(f"-> ROI Point {len(self.clicked_points)}: {(x,y)}")
                elif self.step == "SELECT_TARGET":
                    self.target_point = (x, y)
                    print(f"-> Target Point: {self.target_point}")

        cv2.namedWindow("Video Analysis")
        cv2.setMouseCallback("Video Analysis", mouse_callback)
        
        print("\n=== SMART DISTANCE VIDEO V2 ===")
        print("B1: Click 4 góc sàn (Theo thứ tự: Top-L -> Top-R -> Bot-R -> Bot-L).")
        print("B2: Nhấn 'c' để Confirm.")
        print("B3: Chọn vật mốc (Target). Nhấn SPACE để chạy.")

        while True:
            display_frame = current_frame.copy()

            # --- SETUP PHASE ---
            if self.step == "SELECT_ROI":
                cv2.putText(display_frame, f"SELECT 4 POINTS ({len(self.clicked_points)}/4)", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                for i, p in enumerate(self.clicked_points):
                    cv2.circle(display_frame, p, 5, (0, 255, 255), -1)
                    if i > 0: cv2.line(display_frame, self.clicked_points[i-1], p, (0, 255, 255), 2)
                if len(self.clicked_points) == 4:
                    cv2.line(display_frame, self.clicked_points[3], self.clicked_points[0], (0, 255, 255), 2)

            elif self.step == "SELECT_TARGET":
                pts = np.array(self.clicked_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], True, (255, 100, 0), 2)
                
                if self.target_point:
                    cv2.circle(display_frame, self.target_point, 6, (0, 0, 255), -1)
                    cv2.putText(display_frame, "TARGET", (self.target_point[0]+10, self.target_point[1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # --- RUNNING PHASE ---
            elif self.step == "RUNNING":
                start_t = time.time()
                ret, raw = cap.read()
                if not ret: break
                
                # 1. Pipeline xử lý ảnh y hệt Setup
                undist = self.smart_undistort(raw)
                current_frame = cv2.resize(undist, (TARGET_W, new_h))
                display_frame = current_frame.copy()

                # 2. Stabilizer Logic
                if self.use_stabilizer and self.stabilizer:
                    new_roi, H_global = self.stabilizer.update(current_frame)
                    self.update_homography(new_roi) # Recalculate Homography
                    
                    # Update Target Point
                    if self.original_target_point and H_global is not None:
                        target_arr = np.array([[self.original_target_point]], dtype=np.float32)
                        new_target = cv2.perspectiveTransform(target_arr, H_global)
                        self.target_point = (int(new_target[0][0][0]), int(new_target[0][0][1]))
                    
                    roi_vis = np.array(new_roi, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [roi_vis], True, (0, 100, 255), 2)

                # 3. AI Inference
                if self.ai_thread.input_queue.empty():
                    self.ai_thread.input_queue.put(current_frame.copy())
                if not self.ai_thread.output_queue.empty():
                    self.latest_detections = self.ai_thread.output_queue.get()

                # 4. Draw & Measure
                cv2.circle(display_frame, self.target_point, 5, (0, 0, 255), -1)
                
                for obj in self.latest_detections:
                    foot = obj['ground_point']
                    bx1, by1, bx2, by2 = obj['box']
                    
                    # Vẽ box
                    color = (0, 255, 255) if not obj['is_far'] else (200, 200, 200)
                    cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), color, 1)
                    cv2.circle(display_frame, foot, 5, (0, 255, 0), -1)
                    
                    # Tính khoảng cách
                    dist = self.calculate_distance_real(self.target_point, foot)
                    
                    cv2.line(display_frame, self.target_point, foot, (0, 255, 255), 2)
                    mid = ((self.target_point[0]+foot[0])//2, (self.target_point[1]+foot[1])//2)
                    cv2.putText(display_frame, f"{dist:.2f}m", mid, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                fps = 1.0 / (time.time() - start_t)
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            cv2.imshow("Video Analysis", display_frame)
            
            wait_time = 30 if self.step != "RUNNING" else 1
            key = cv2.waitKey(wait_time)
            if key == ord('q'): break
            
            # --- Key Handlers ---
            if self.step == "SELECT_ROI":
                if key == ord('r'): self.clicked_points = []
                if key == ord('c'):
                    if len(self.clicked_points) == 4:
                        print("-> Init Stabilizer & Homography...")
                        self.update_homography(self.clicked_points)
                        if self.stabilizer:
                            self.stabilizer.initialize(current_frame, self.clicked_points)
                        self.step = "SELECT_TARGET"

            elif self.step == "SELECT_TARGET":
                if key == ord(' '):
                    if self.target_point:
                        self.original_target_point = self.target_point
                        self.step = "RUNNING"

        self.ai_thread.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = VideoDistanceApp()
    app.run_video()