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

try:
    # Import class mới PerspectiveStabilizer
    from stabilizer import PerspectiveStabilizer
except ImportError:
    print("[WARN] Thiếu stabilizer.py")
    PerspectiveStabilizer = None

# ================= CẤU HÌNH =================
VIDEO_PATH = r'../test_imgs/cam_2/cam_2.mp4' 
CALIB_FILE = 'calibration.json' 
CONFIG_FILE = 'config.json'
TARGET_W = 1200 

# ================= CLASS YOLO (GIỮ NGUYÊN) =================
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
        print(f"[AI] Loading Model...")
        self.model = YOLO(self.model_path)
        while not self.stopped:
            try:
                frame = self.input_queue.get(timeout=0.1)
                if self.model:
                    results = self.model(frame, verbose=False, device=self.device)
                    processed = self.process_results(results, frame.shape[1], frame.shape[0])
                    if not self.output_queue.empty():
                        try: self.output_queue.get_nowait()
                        except: pass
                    self.output_queue.put(processed)
                self.input_queue.task_done()
            except queue.Empty: continue
            except Exception as e: print(f"[AI] Error: {e}")

    def process_results(self, results, img_w, img_h):
        detected = []
        RIGHT_MARGIN = 80
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                if x2 > (img_w - RIGHT_MARGIN): continue
                h_box = y2 - y1
                # Logic phân loại xa/gần
                is_far = h_box < 60
                ground_point = None
                if not is_far and kpts is not None and len(kpts) > i:
                    kp = kpts[i]
                    if kp[15][2] > 0.5 or kp[16][2] > 0.5:
                        ground_point = (int((kp[15][0]+kp[16][0])/2), int((kp[15][1]+kp[16][1])/2))
                if ground_point is None: ground_point = (int((x1+x2)/2), y2)
                detected.append({'box': box, 'ground_point': ground_point, 'is_far': is_far})
        return detected

    def stop(self): self.stopped = True

# ================= MAIN APP (SỬA LOGIC CỐT LÕI) =================
class VideoDistanceApp:
    def __init__(self):
        self.matrix_homography_static = None # Homography Tĩnh (Frame 0 -> Real World)
        self.scale_px_per_meter = 1.0
        self.real_world = {}
        
        self.step = "SELECT_ROI"
        self.clicked_points = []
        self.target_point_static = None # Tọa độ Target trên Frame Gốc (TĨNH)
        
        # Stabilizer Mới
        self.stabilizer = PerspectiveStabilizer() if PerspectiveStabilizer else None
        self.M_curr = np.eye(3, dtype=np.float32) # Ma trận biến đổi Frame Gốc -> Hiện tại

        self.calib_data = self.load_json(CALIB_FILE)
        self.load_config(CONFIG_FILE)
        
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.ai_thread = YOLOThread('../weights/yolo11n-pose.onnx', self.device)
        self.ai_thread.daemon = True
        self.ai_thread.start()
        self.latest_detections = []

    def load_json(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f: return json.load(f)
        return None

    def load_config(self, path):
        data = self.load_json(path)
        if data:
            self.real_world = data['real_world']
            self.scale_px_per_meter = data['settings'].get('scale_px_per_meter', 1.0)

    def smart_undistort(self, img):
        if not self.calib_data: return img
        try:
            h, w = img.shape[:2]
            K = np.array(self.calib_data['camera_matrix'])
            D = np.array(self.calib_data['distortion_coefficients'])
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
            return cv2.undistort(img, K, D, None, new_K)
        except: return img

    # Hàm tính toán Real World Coords (Giữ nguyên logic hình học của bạn)
    def get_real_coords_from_params(self):
        rw = self.real_world
        l1, l2, l3, l4, d13 = rw.get('L1',0), rw.get('L2',0), rw.get('L3',0), rw.get('L4',0), rw.get('diag_13',0)
        if l1==0: return None
        p1=(0.0,0.0); p2=(l1,0.0)
        try:
            cos_a = (l1**2 + d13**2 - l2**2)/(2*l1*d13)
            cos_a = max(-1.0, min(1.0, cos_a))
            alpha = math.acos(cos_a)
            p3 = (d13 * math.cos(alpha), d13 * math.sin(alpha))
            d = d13
            a_val = (l4**2 - l3**2 + d**2) / (2*d)
            h_val = math.sqrt(max(0, l4**2 - a_val**2))
            x2, y2 = p3
            x0 = p1[0] + a_val * (x2 - p1[0]) / d
            y0 = p1[1] + a_val * (y2 - p1[1]) / d
            rx = -(y2 - p1[1]) / d; ry = (x2 - p1[0]) / d
            p4_a = (x0 + h_val * rx, y0 + h_val * ry)
            p4_b = (x0 - h_val * rx, y0 - h_val * ry)
            
            def cross_product(o, a, b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
            p4 = p4_a if cross_product(p1, p3, p4_a) < 0 else p4_b
            return [p1, p2, p3, p4]
        except: return None

    def compute_homography_static(self):
        real_coords = self.get_real_coords_from_params()
        if real_coords and len(self.clicked_points) == 4:
            dst_pts = np.float32([[pt[0]*self.scale_px_per_meter, pt[1]*self.scale_px_per_meter] for pt in real_coords])
            src_pts = np.float32(self.clicked_points)
            try:
                self.matrix_homography_static = cv2.getPerspectiveTransform(src_pts, dst_pts)
                print("[INFO] Static Homography Calculated.")
                return True
            except: pass
        return False

    # HÀM QUAN TRỌNG: Transform điểm từ hệ Gốc -> hệ Hiện tại (để vẽ)
    def transform_point_forward(self, point_static):
        # Point Static (x, y) -> Point Current (x', y')
        # Công thức: P' = M * P
        if point_static is None: return None
        px, py = point_static
        vec = np.array([[[px, py]]], dtype=np.float32)
        trans_vec = cv2.perspectiveTransform(vec, self.M_curr)
        return (int(trans_vec[0][0][0]), int(trans_vec[0][0][1]))

    # HÀM QUAN TRỌNG: Transform điểm từ hệ Hiện tại -> hệ Gốc (để tính toán)
    def transform_point_inverse(self, point_current):
        # Point Current (x', y') -> Point Static (x, y)
        # Công thức: P = inv(M) * P'
        px, py = point_current
        vec = np.array([[[px, py]]], dtype=np.float32)
        try:
            # Tính nghịch đảo ma trận M
            M_inv = np.linalg.inv(self.M_curr)
            trans_vec = cv2.perspectiveTransform(vec, M_inv)
            return (trans_vec[0][0][0], trans_vec[0][0][1])
        except:
            return point_current # Fallback nếu ma trận không nghịch đảo được

    def calculate_distance_final(self, p_target_static, p_foot_current):
        """
        p_target_static: Tọa độ gốc (lúc setup)
        p_foot_current: Tọa độ chân ở frame hiện tại (đang bị rung/lệch)
        """
        if self.matrix_homography_static is None: return 0.0
        
        # B1. Đưa chân người về hệ tọa độ gốc
        p_foot_static = self.transform_point_inverse(p_foot_current)
        
        # B2. Tính khoảng cách trong hệ tọa độ gốc (Dùng Homography Tĩnh)
        pts = np.float32([p_target_static, p_foot_static]).reshape(-1, 1, 2)
        world_pts = cv2.perspectiveTransform(pts, self.matrix_homography_static)
        
        dx = world_pts[0][0][0] - world_pts[1][0][0]
        dy = world_pts[0][0][1] - world_pts[1][0][1]
        dist_px = math.sqrt(dx*dx + dy*dy)
        
        return dist_px / self.scale_px_per_meter

    def run_video(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened(): return
        
        ret, frame = cap.read()
        frame = self.smart_undistort(frame)
        h, w = frame.shape[:2]
        scale = TARGET_W / w
        new_h = int(h * scale)
        frame = cv2.resize(frame, (TARGET_W, new_h))
        
        def mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.step == "SELECT_ROI" and len(self.clicked_points) < 4:
                    self.clicked_points.append((x, y))
                elif self.step == "SELECT_TARGET":
                    self.target_point_static = (x, y) # Lưu tọa độ gốc

        cv2.namedWindow("App")
        cv2.setMouseCallback("App", mouse)

        while True:
            display = frame.copy()

            if self.step == "RUNNING":
                ret, raw = cap.read()
                if not ret: break
                
                curr = cv2.resize(self.smart_undistort(raw), (TARGET_W, new_h))
                display = curr.copy()
                
                # 1. UPDATE STABILIZER -> Lấy Ma trận M
                if self.stabilizer:
                    self.M_curr = self.stabilizer.update(curr)

                # 2. AI DETECT
                if self.ai_thread.input_queue.empty(): self.ai_thread.input_queue.put(curr.copy())
                if not self.ai_thread.output_queue.empty(): self.latest_detections = self.ai_thread.output_queue.get()
                
                # 3. VISUALIZATION (Vẽ)
                # Target phải được "biến hình" theo M để dính vào mặt đường trên frame hiện tại
                target_vis = self.transform_point_forward(self.target_point_static)
                if target_vis:
                    cv2.circle(display, target_vis, 6, (0,0,255), -1)
                
                # Vẽ lại ROI (để thấy độ rung của sàn)
                if len(self.clicked_points) == 4:
                     roi_arr = np.array([self.clicked_points], dtype=np.float32)
                     roi_trans = cv2.perspectiveTransform(roi_arr, self.M_curr)
                     cv2.polylines(display, [np.int32(roi_trans)], True, (0, 100, 255), 1)

                for obj in self.latest_detections:
                    foot_curr = obj['ground_point']
                    box = obj['box']
                    
                    # 4. MEASUREMENT (Đo)
                    # Input: Target Gốc & Chân Hiện tại
                    dist = self.calculate_distance_final(self.target_point_static, foot_curr)
                    
                    # Vẽ vời
                    cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0,255,255), 1)
                    cv2.circle(display, foot_curr, 5, (0,255,0), -1)
                    if target_vis:
                        cv2.line(display, target_vis, foot_curr, (0,255,255), 2)
                        mid = ((target_vis[0]+foot_curr[0])//2, (target_vis[1]+foot_curr[1])//2)
                        cv2.putText(display, f"{dist:.2f}m", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # ... (Các bước SELECT_ROI vẽ tĩnh) ...
            if self.step == "SELECT_ROI":
                for p in self.clicked_points: cv2.circle(display, p, 5, (0,255,0), -1)
            elif self.step == "SELECT_TARGET" and self.target_point_static:
                cv2.circle(display, self.target_point_static, 6, (0,0,255), -1)
            
            cv2.imshow("App", display)
            k = cv2.waitKey(1)
            if k == ord('q'): break
            
            if self.step == "SELECT_ROI" and k == ord('c') and len(self.clicked_points)==4:
                self.compute_homography_static()
                # Init Stabilizer: Truyền roi_points để nó biết đường mà né
                if self.stabilizer: self.stabilizer.initialize(frame, self.clicked_points)
                self.step = "SELECT_TARGET"
            
            elif self.step == "SELECT_TARGET" and k == ord(' ') and self.target_point_static:
                self.step = "RUNNING"
                
        self.ai_thread.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    VideoDistanceApp().run_video()