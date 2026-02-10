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

# Import module stabilizer vừa tạo
try:
    from stabilizer import ROIStabilizer
except ImportError:
    print("[WARN] Không tìm thấy file stabilizer.py. Chống rung sẽ bị tắt.")
    ROIStabilizer = None

# ================= CẤU HÌNH HỆ THỐNG =================
# Bạn hãy sửa lại đường dẫn cho đúng với máy của bạn
VIDEO_PATH = r'../test_imgs/cam_2/cam_2.mp4' 
CALIB_FILE = r'../calibration.json' # Nếu không có thì để None
CONFIG_FILE = 'config.json'
TARGET_W = 1200  # Resize chiều ngang để xử lý nhanh hơn

# ================= CLASS XỬ LÝ AI (ĐA LUỒNG) =================
class YOLOThread(threading.Thread):
    def __init__(self, model_path, device):
        threading.Thread.__init__(self)
        self.model_path = model_path
        self.device = device
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.stopped = False
        self.model = None

    def run(self):
        print(f"[AI] Loading YOLO model on {self.device}...")
        try:
            self.model = YOLO(self.model_path)
            # Warmup model
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, verbose=False, device=self.device)
            print("[AI] Model Ready.")
        except Exception as e:
            print(f"[AI ERR] Không load được model: {e}")
            self.stopped = True

        while not self.stopped:
            try:
                frame = self.input_queue.get(timeout=0.1)
                if self.model:
                    results = self.model(frame, verbose=False, device=self.device)
                    processed = self.process_results(results, frame.shape[1], frame.shape[0])
                    
                    # Xóa kết quả cũ nếu chưa kịp lấy để tránh delay
                    if not self.output_queue.empty():
                        try: self.output_queue.get_nowait()
                        except queue.Empty: pass
                    self.output_queue.put(processed)
                self.input_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[AI LOOP ERR] {e}")

    def process_results(self, results, img_w, img_h):
        detected_objects = []
        RIGHT_MARGIN = 80 # Lọc viền đen bên phải (nếu có do undistort)
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            keypoints = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                
                # Lọc nhiễu kích thước và vị trí
                if x2 > (img_w - RIGHT_MARGIN): continue
                if (x2 - x1) < 20 or (y2 - y1) < 20: continue
                
                h_box = y2 - y1
                # Nếu người quá nhỏ (< 60px), coi là ở xa -> Dùng đáy Box thay vì Keypoint
                IS_FAR_AWAY = h_box < 60
                
                ground_point = None
                
                if not IS_FAR_AWAY and keypoints is not None and len(keypoints) > i:
                    kpts = keypoints[i]
                    left_ankle = kpts[15]  # Mắt cá trái
                    right_ankle = kpts[16] # Mắt cá phải
                    
                    l_conf, r_conf = left_ankle[2], right_ankle[2]

                    if l_conf > 0.5 and r_conf > 0.5:
                        gx = int((left_ankle[0] + right_ankle[0]) / 2)
                        gy = int((left_ankle[1] + right_ankle[1]) / 2)
                        ground_point = (gx, gy)
                    elif l_conf > 0.5:
                        ground_point = (int(left_ankle[0]), int(left_ankle[1]))
                    elif r_conf > 0.5:
                        ground_point = (int(right_ankle[0]), int(right_ankle[1]))

                # Fallback: Dùng điểm giữa cạnh đáy của Box
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
        # Biến hình học
        self.matrix_homography = None # Ma trận biến đổi (TĨNH)
        self.scale_px_per_meter = 1.0
        self.real_world_config = {} 
        
        # Trạng thái ứng dụng
        self.step = "SELECT_ROI"
        self.clicked_points = [] 
        self.target_point = None  # Điểm đích (TĨNH - Pixel gốc)
        
        # Biến chống rung
        self.stabilizer = ROIStabilizer() if ROIStabilizer else None
        self.use_stabilizer = True
        self.total_dx = 0.0 # Tổng độ lệch X tích lũy
        self.total_dy = 0.0 # Tổng độ lệch Y tích lũy

        # Load dữ liệu
        self.calib_data = self.load_json(CALIB_FILE)
        self.load_config(CONFIG_FILE)
        
        # Khởi động AI
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        # Thay đường dẫn weights nếu cần
        self.ai_thread = YOLOThread('../weights/yolo11n-pose.onnx', self.device)
        self.ai_thread.daemon = True
        self.ai_thread.start()
        
        self.latest_detections = []

    def load_json(self, path):
        if os.path.exists(path):
            try:
                with open(path, 'r') as f: return json.load(f)
            except: pass
        return None

    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                self.real_world_config = data['real_world']
                self.scale_px_per_meter = data['settings'].get('scale_px_per_meter', 1.0)
                print(f"[CONFIG] Loaded Real World Params: {self.real_world_config}")
        except Exception as e: 
            print(f"[ERR] Config Load Error: {e}")

    # Hàm Undistort thông minh (giữ nguyên từ code cũ)
    def smart_undistort(self, img):
        if self.calib_data is None: return img
        try:
            h_curr, w_curr = img.shape[:2]
            K = np.array(self.calib_data['camera_matrix'])
            D = np.array(self.calib_data['distortion_coefficients'])
            
            # Tính toán lại K nếu kích thước video khác kích thước calib
            if 'image_resolution' in self.calib_data:
                calib_w, calib_h = self.calib_data['image_resolution']
                if w_curr != calib_w or h_curr != calib_h:
                    scale_x = w_curr / calib_w
                    scale_y = h_curr / calib_h
                    K[0, 0] *= scale_x; K[1, 1] *= scale_y
                    K[0, 2] *= scale_x; K[1, 2] *= scale_y
            
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w_curr, h_curr), 1, (w_curr, h_curr))
            map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w_curr, h_curr), 5)
            return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        except:
            return img

    # Tính toán tọa độ thực tế của 4 điểm ROI từ file config (L1, L2, L3, L4...)
    def get_real_coords_from_params(self):
        rw = self.real_world_config
        l1 = rw.get('L1', 0)
        l2 = rw.get('L2', 0)
        l3 = rw.get('L3', 0)
        l4 = rw.get('L4', 0)
        d13 = rw.get('diag_13', 0)

        if l1 == 0: return None

        # P1 là gốc (0,0)
        p1 = (0.0, 0.0)
        # P2 nằm trên trục hoành
        p2 = (l1, 0.0)

        try:
            # Tính P3 bằng định lý hàm Cos
            cos_a = (l1**2 + d13**2 - l2**2) / (2 * l1 * d13)
            cos_a = max(-1.0, min(1.0, cos_a)) # Clip giá trị để tránh lỗi math domain
            alpha = math.acos(cos_a)
            p3 = (d13 * math.cos(alpha), d13 * math.sin(alpha))
            
            # Tính P4 (Giao điểm của 2 đường tròn bán kính L4 và L3)
            d = d13 # Khoảng cách P1-P3
            # a là khoảng cách từ P1 đến hình chiếu của P4 lên P1P3
            a_val = (l4**2 - l3**2 + d**2) / (2 * d)
            h_val = math.sqrt(max(0, l4**2 - a_val**2)) # Chiều cao
            
            x2, y2 = p3
            x0 = p1[0] + a_val * (x2 - p1[0]) / d
            y0 = p1[1] + a_val * (y2 - p1[1]) / d
            
            rx = -(y2 - p1[1]) / d
            ry = (x2 - p1[0]) / d
            
            p4_a = (x0 + h_val * rx, y0 + h_val * ry)
            p4_b = (x0 - h_val * rx, y0 - h_val * ry)
            
            # Kiểm tra hướng vector để chọn P4 đúng (Lồi/Lõm)
            def cross_product(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

            if cross_product(p1, p3, p4_a) < 0: 
                p4 = p4_a 
            else: 
                p4 = p4_b
            
            return [p1, p2, p3, p4]
        except Exception as e:
            print(f"[MATH ERR] Lỗi tính toán hình học: {e}")
            return None

    # Hàm tính Homography TĨNH (Chỉ chạy 1 lần)
    def compute_homography_static(self):
        real_coords = self.get_real_coords_from_params()
        if real_coords and len(self.clicked_points) == 4:
            # Scale tọa độ thực ra pixel (để dễ debug nếu cần hiển thị map 2D)
            dst_pts = np.float32([[pt[0] * self.scale_px_per_meter, pt[1] * self.scale_px_per_meter] for pt in real_coords])
            src_pts = np.float32(self.clicked_points)
            
            try:
                self.matrix_homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
                print("[INFO] Homography Matrix Calculated (STATIC).")
                return True
            except Exception as e:
                print(f"[ERR] Homography Failed: {e}")
        return False

    # Hàm tính khoảng cách chuẩn (Đã khử rung)
    def calculate_distance_corrected(self, p_target_original, p_foot_corrected):
        """
        p_target_original: Tọa độ target trên frame gốc.
        p_foot_corrected: Tọa độ chân đã được trừ đi độ rung (đưa về frame gốc).
        """
        if self.matrix_homography is None: return 0.0
        
        pts = np.float32([p_target_original, p_foot_corrected]).reshape(-1, 1, 2)
        
        # Biến đổi sang hệ tọa độ thực (Real World)
        world_pts = cv2.perspectiveTransform(pts, self.matrix_homography)
        
        # Tính khoảng cách Euclidean
        dx = world_pts[0][0][0] - world_pts[1][0][0]
        dy = world_pts[0][0][1] - world_pts[1][0][1]
        dist_px = math.sqrt(dx*dx + dy*dy)
        
        return dist_px / self.scale_px_per_meter

    def run_video(self):
        if not os.path.exists(VIDEO_PATH):
            print(f"[ERR] Video path not found: {VIDEO_PATH}")
            return

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened(): return

        # --- SETUP KHUNG HÌNH ĐẦU TIÊN ---
        ret, first_frame = cap.read()
        if not ret: return
        
        # Undistort & Resize
        undist = self.smart_undistort(first_frame)
        h_orig, w_orig = undist.shape[:2]
        scale_factor = TARGET_W / w_orig
        new_h = int(h_orig * scale_factor)
        
        current_frame = cv2.resize(undist, (TARGET_W, new_h))
        
        # --- MOUSE CALLBACK ---
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.step == "SELECT_ROI":
                    if len(self.clicked_points) < 4:
                        self.clicked_points.append((x, y))
                        print(f"-> Clicked ROI Point: {(x,y)}")
                elif self.step == "SELECT_TARGET":
                    self.target_point = (x, y)
                    print(f"-> Target Point Selected: {self.target_point}")

        cv2.namedWindow("Distance Measurement App")
        cv2.setMouseCallback("Distance Measurement App", mouse_callback)

        print("\n=== HƯỚNG DẪN ===")
        print("1. Click 4 điểm sàn để tạo ROI (Điểm 1 sẽ là ĐIỂM NEO chống rung).")
        print("2. Nhấn 'c' để xác nhận ROI.")
        print("3. Click vào vật mốc (Target).")
        print("4. Nhấn SPACE để bắt đầu chạy.")

        while True:
            display_frame = current_frame.copy()

            # --- GIAI ĐOẠN 1: CHỌN ROI ---
            if self.step == "SELECT_ROI":
                for i, p in enumerate(self.clicked_points):
                    # Điểm đầu tiên (Anchor) màu đỏ, còn lại màu vàng
                    color = (0, 0, 255) if i == 0 else (0, 255, 255) 
                    cv2.circle(display_frame, p, 5, color, -1)
                    if i > 0: cv2.line(display_frame, self.clicked_points[i-1], p, (0, 255, 255), 2)
                
                if len(self.clicked_points) == 4:
                     cv2.line(display_frame, self.clicked_points[3], self.clicked_points[0], (0, 255, 255), 2)
                     cv2.putText(display_frame, "Press 'c' to Confirm ROI", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # --- GIAI ĐOẠN 2: CHỌN TARGET ---
            elif self.step == "SELECT_TARGET":
                pts = np.array(self.clicked_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], True, (255, 100, 0), 2)
                
                if self.target_point:
                    cv2.circle(display_frame, self.target_point, 6, (0, 0, 255), -1)
                    cv2.putText(display_frame, "Press SPACE to Run", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # --- GIAI ĐOẠN 3: CHẠY VIDEO ---
            elif self.step == "RUNNING":
                ret, raw = cap.read()
                if not ret: break # Hết video hoặc lỗi
                
                # Tiền xử lý (Phải giống hệt lúc setup)
                frame_undist = self.smart_undistort(raw)
                current_frame = cv2.resize(frame_undist, (TARGET_W, new_h))
                display_frame = current_frame.copy()

                # 1. CẬP NHẬT STABILIZER (Tính độ trôi)
                if self.use_stabilizer and self.stabilizer:
                    # dx, dy là độ lệch frame hiện tại so với frame trước
                    dx, dy = self.stabilizer.update(current_frame)
                    
                    # Cộng dồn vào tổng độ lệch
                    self.total_dx += dx
                    self.total_dy += dy
                    
                    # [Debug] Vẽ điểm Anchor hiện tại (nó sẽ di chuyển nếu cam rung)
                    if self.stabilizer.anchor_point:
                        # Anchor gốc + trôi = Anchor trên màn hình
                        curr_anchor_vis = (int(self.stabilizer.anchor_point[0] + self.total_dx),
                                           int(self.stabilizer.anchor_point[1] + self.total_dy))
                        cv2.circle(display_frame, curr_anchor_vis, 5, (0, 0, 255), -1) # Chấm đỏ chạy theo rung
                        cv2.circle(display_frame, self.stabilizer.anchor_point, 5, (0, 255, 0), 1) # Vòng tròn gốc đứng yên

                # 2. GỬI ẢNH CHO AI DETECT
                if self.ai_thread.input_queue.empty():
                    self.ai_thread.input_queue.put(current_frame.copy())
                
                # 3. NHẬN KẾT QUẢ TỪ AI
                if not self.ai_thread.output_queue.empty():
                    self.latest_detections = self.ai_thread.output_queue.get()

                # 4. VẼ VÀ ĐO KHOẢNG CÁCH
                # Vẽ Target Point (phải cộng drift để hiển thị đúng vị trí trên màn hình rung)
                target_vis_x = int(self.target_point[0] + self.total_dx)
                target_vis_y = int(self.target_point[1] + self.total_dy)
                target_vis = (target_vis_x, target_vis_y)
                
                cv2.circle(display_frame, target_vis, 6, (0, 0, 255), -1)

                for obj in self.latest_detections:
                    foot_curr = obj['ground_point'] # Tọa độ chân trên frame hiện tại (đang bị rung)
                    box = obj['box']
                    
                    # --- CORE LOGIC: KHỬ RUNG ---
                    # Muốn tính khoảng cách trên hệ tọa độ gốc, ta phải trừ đi độ lệch
                    # Foot_Corrected = Foot_Current - Total_Drift
                    foot_corrected_x = foot_curr[0] - self.total_dx
                    foot_corrected_y = foot_curr[1] - self.total_dy
                    foot_corrected = (foot_corrected_x, foot_corrected_y)
                    
                    # Tính khoảng cách dùng Homography tĩnh
                    dist = self.calculate_distance_corrected(self.target_point, foot_corrected)
                    
                    # Vẽ dây nối (trên màn hình hiện tại)
                    cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 1)
                    cv2.circle(display_frame, foot_curr, 5, (0, 255, 0), -1)
                    cv2.line(display_frame, target_vis, foot_curr, (0, 255, 255), 2)
                    
                    # Hiển thị text
                    mid_x = (target_vis[0] + foot_curr[0]) // 2
                    mid_y = (target_vis[1] + foot_curr[1]) // 2
                    cv2.putText(display_frame, f"{dist:.2f}m", (mid_x, mid_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Hiển thị frame
            cv2.imshow("Distance Measurement App", display_frame)
            
            # Xử lý phím bấm
            wait_time = 30 if self.step != "RUNNING" else 1
            key = cv2.waitKey(wait_time)
            
            if key == ord('q'): 
                break
            
            # Logic chuyển trạng thái
            if self.step == "SELECT_ROI":
                if key == ord('r'): # Reset points
                    self.clicked_points = []
                elif key == ord('c') and len(self.clicked_points) == 4:
                    print("-> Computing Homography & Initializing Stabilizer...")
                    
                    # 1. Tính Homography Tĩnh
                    if self.compute_homography_static():
                        # 2. Init Stabilizer tại điểm click đầu tiên (Anchor)
                        if self.stabilizer:
                            anchor = self.clicked_points[0] # Tuple (x, y)
                            self.stabilizer.initialize(current_frame, anchor)
                        
                        self.step = "SELECT_TARGET"
                    else:
                        print("[ERR] Không thể tính Homography. Kiểm tra lại config.")

            elif self.step == "SELECT_TARGET":
                if key == ord(' ') and self.target_point:
                    print("-> Start Running Video...")
                    self.step = "RUNNING"

        # Cleanup
        self.ai_thread.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = VideoDistanceApp()
    app.run_video()