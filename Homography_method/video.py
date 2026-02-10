import cv2
import numpy as np
import math
import os
import json
from ultralytics import YOLO
import torch
import time

# IMPORT MODULE CŨ
from height_estimator import HeightEstimator

# ================= CẤU HÌNH =================
VIDEO_PATH = '..\\test_imgs\\cam_2\\cam_2.mp4' 
CALIB_FILE = '..\\calibration.json'
CONFIG_FILE = 'config.json'
TARGET_W = 1200 

class VideoDistanceApp:
    def __init__(self):
        # --- CẤU HÌNH MODE ---
        self.mode = "DISTANCE"
        self.paused = False
        
        # --- TRACKING ---
        self.target_tracker = None      # Tracker cho điểm đích (Target)
        self.roi_trackers = []          # List 4 Tracker cho 4 góc ROI
        self.roi_points_curr = []       # Tọa độ hiện tại của 4 góc ROI (sẽ thay đổi liên tục)
        self.target_point = None        # Tọa độ hiện tại của Target
        self.tracking_initialized = False # Cờ báo đã khởi tạo tracking chưa
        
        # Tools
        self.height_tool = HeightEstimator()
        self.yolo_model = None
        
        # Data & Calibration
        self.real_world = {}
        self.cam_real_pos = (0.5, -18.0)
        self.clicked_points_orig = []   # 4 điểm gốc từ config (chỉ dùng để init)
        self.matrix_homography = None
        self.scale_px_per_meter = 1.0 
        self.map1, self.map2 = None, None
        
        # Runtime
        self.current_frame = None
        self.detected_objects = []

        # Setup Device
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 0
            print(f"[INFO] GPU Activated: {torch.cuda.get_device_name(0)}")
        
        try:
            self.yolo_model = YOLO('..\\weights\\yolo11n-pose.onnx') 
            print("[INFO] YOLO Model loaded.")
        except Exception as e:
            print(f"[ERR] Load YOLO failed: {e}")

    def init_calibration_maps(self, frame_size):
        if not os.path.exists(CALIB_FILE): return
        try:
            with open(CALIB_FILE, 'r') as f: data = json.load(f)
            K = np.array(data['camera_matrix'])
            D = np.array(data['distortion_coefficients'])
            w_curr, h_curr = frame_size
            if 'image_resolution' in data:
                calib_w, calib_h = data['image_resolution']
                if w_curr != calib_w or h_curr != calib_h:
                    scale_x = w_curr / calib_w; scale_y = h_curr / calib_h
                    K[0, 0] *= scale_x; K[1, 1] *= scale_y; K[0, 2] *= scale_x; K[1, 2] *= scale_y
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w_curr, h_curr), 1, (w_curr, h_curr))
            self.map1, self.map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w_curr, h_curr), 5)
            self.height_tool.load_focal_length(CALIB_FILE, TARGET_W)
        except Exception as e:
            print(f"[ERR] Calibration Init: {e}")

    def load_config(self):
        if not os.path.exists(CONFIG_FILE): return False
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                self.real_world = data['real_world']
                self.cam_real_pos = (data['camera']['real_x'], data['camera']['real_y'])
                self.scale_px_per_meter = data['settings'].get('scale_px_per_meter', 1.0)
                if 'points_px' in data:
                    self.clicked_points_orig = [tuple(p) for p in data['points_px']]
                    # Ban đầu thì điểm hiện tại = điểm gốc
                    self.roi_points_curr = list(self.clicked_points_orig)
                    # Tính Homography lần đầu
                    self.compute_homography(self.roi_points_curr)
                    return True
        except: return False

    def get_quadrilateral_coords(self, l1, l2, l3, l4, d13):
        # Hàm tính toán hình học thực tế (giữ nguyên)
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

    def compute_homography(self, current_pixel_points):
        """
        Tính lại Matrix Homography dựa trên 4 điểm pixel HIỆN TẠI (đã bị dịch chuyển)
        so với 4 điểm thực tế (cố định).
        """
        if len(current_pixel_points) < 4: return
        rw = self.real_world
        real_coords = self.get_quadrilateral_coords(rw['L1'], rw['L2'], rw['L3'], rw['L4'], rw['diag_13'])
        if not real_coords: return
        
        # Mapping: Pixel mới -> Real World cũ
        dst_pts = np.float32([[pt[0]*self.scale_px_per_meter, pt[1]*self.scale_px_per_meter] for pt in real_coords])
        src_pts = np.float32(current_pixel_points)
        
        try:
            self.matrix_homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
        except:
            pass # Tránh crash nếu tracking bị lỗi gây ra shape kỳ dị

    def calculate_distance_points(self, p1, p2):
        if self.matrix_homography is None: return 0.0
        pts = np.float32([p1, p2]).reshape(-1, 1, 2)
        trans_pts = cv2.perspectiveTransform(pts, self.matrix_homography)
        dist_px = np.linalg.norm(trans_pts[0][0] - trans_pts[1][0])
        return dist_px / self.scale_px_per_meter

    # --- KHỞI TẠO TRACKER CHO 4 ĐIỂM ROI ---
    def init_roi_tracking(self, frame):
        self.roi_trackers = []
        for pt in self.clicked_points_orig:
            # Tạo vùng box nhỏ quanh điểm góc để track
            x, y = pt
            box_size = 30
            bbox = (x - box_size//2, y - box_size//2, box_size, box_size)
            
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, bbox)
            self.roi_trackers.append(tracker)
        
        self.tracking_initialized = True
        print("[TRACKING] Đã khởi tạo theo dõi 4 góc ROI.")

    # --- CẬP NHẬT TRACKER (ROI + TARGET) ---
    def update_all_trackers(self, frame):
        # 1. Update ROI Trackers
        new_roi_points = []
        if self.tracking_initialized:
            for i, tracker in enumerate(self.roi_trackers):
                success, box = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in box]
                    center = (x + w//2, y + h//2)
                    new_roi_points.append(center)
                else:
                    # Nếu mất dấu, dùng lại điểm cũ
                    new_roi_points.append(self.roi_points_curr[i])
            
            # Cập nhật tọa độ ROI mới và tính lại Homography
            self.roi_points_curr = new_roi_points
            self.compute_homography(self.roi_points_curr)

        # 2. Update Target Point Tracker (nếu có)
        track_box = None
        if self.target_tracker is not None:
            success, box = self.target_tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                self.target_point = (x + w//2, y + h//2)
                track_box = (x, y, w, h)
            else:
                self.target_tracker = None # Mất dấu thì hủy luôn

        return track_box

    def process_frame(self, raw_frame):
        # 1. Pre-processing
        if self.map1 is not None:
            frame = cv2.remap(raw_frame, self.map1, self.map2, cv2.INTER_LINEAR)
        else: frame = raw_frame
        
        h, w = frame.shape[:2]
        scale = TARGET_W / w
        new_h = int(h * scale)
        frame_resized = cv2.resize(frame, (TARGET_W, new_h))
        
        # 2. Khởi tạo Tracking ROI ở frame đầu tiên
        if not self.tracking_initialized and len(self.clicked_points_orig) == 4:
            self.init_roi_tracking(frame_resized)
            
        # 3. Update vị trí các điểm (ROI + Target)
        target_box = self.update_all_trackers(frame_resized)
        
        # 4. Detect Object
        self.detect_objects(frame_resized)
        
        return frame_resized, target_box

    def detect_objects(self, img):
        if self.yolo_model is None: return
        results = self.yolo_model(img, verbose=False, device=self.device, conf=0.5)
        self.detected_objects = []
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            kpts_data = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                ground_point = (int((x1+x2)/2), y2)
                head_point = (int((x1+x2)/2), y1)
                
                if kpts_data is not None and len(kpts_data) > i:
                    kp = kpts_data[i]
                    if kp[15][2] > 0.5 and kp[16][2] > 0.5:
                        ground_point = (int((kp[15][0]+kp[16][0])/2), int((kp[15][1]+kp[16][1])/2))
                    if kp[0][2] > 0.5:
                        head_point = (int(kp[0][0]), int(kp[0][1]))

                obj_info = {'box': box, 'head': head_point, 'foot': ground_point, 'h_real': 0.0, 'd_to_target': 0.0}
                
                # Tính toán dựa trên Homography MỚI NHẤT
                if self.mode == "HEIGHT":
                    h_real, _ = self.height_tool.calculate(head_point, ground_point, self.matrix_homography, self.cam_real_pos)
                    obj_info['h_real'] = h_real
                elif self.mode == "DISTANCE" and self.target_point:
                    d_target = self.calculate_distance_points(ground_point, self.target_point)
                    obj_info['d_to_target'] = d_target
                
                self.detected_objects.append(obj_info)

    def draw_overlays(self, img, target_box):
        # 1. Vẽ ROI (Sử dụng self.roi_points_curr để thấy nó di chuyển)
        if len(self.roi_points_curr) == 4:
            pts = np.array(self.roi_points_curr, np.int32).reshape((-1, 1, 2))
            # Vẽ màu đỏ đậm hơn để dễ nhìn
            cv2.polylines(img, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
            for pt in self.roi_points_curr:
                cv2.circle(img, pt, 3, (0, 0, 255), -1)

        # 2. Vẽ Target Point
        if self.mode == "DISTANCE" and self.target_point:
            cv2.circle(img, self.target_point, 5, (0, 255, 255), -1)
            cv2.circle(img, self.target_point, 12, (0, 255, 255), 2)
            if target_box:
                tx, ty, tw, th = target_box
                cv2.rectangle(img, (tx, ty), (tx+tw, ty+th), (0, 255, 255), 1)

        # 3. Vẽ Object Info
        for obj in self.detected_objects:
            x1, y1, x2, y2 = obj['box']
            foot = obj['foot']
            cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), 1)

            if self.mode == "HEIGHT":
                cv2.line(img, obj['head'], foot, (0, 255, 0), 2)
                cv2.putText(img, f"H: {obj['h_real']:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif self.mode == "DISTANCE":
                cv2.circle(img, foot, 4, (255, 0, 0), -1)
                if self.target_point:
                    cv2.line(img, foot, self.target_point, (0, 165, 255), 2)
                    mid = ((foot[0]+self.target_point[0])//2, (foot[1]+self.target_point[1])//2)
                    cv2.putText(img, f"{obj['d_to_target']:.2f}m", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # 4. UI Status
        status_text = "PAUSED" if self.paused else "PLAYING"
        cv2.rectangle(img, (0, 0), (TARGET_W, 60), (0, 0, 0), -1)
        cv2.putText(img, f"MODE: {self.mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(img, status_text, (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if self.paused else (0, 255, 0), 2)
        cv2.putText(img, "AUTO ROI TRACKING ACTIVE", (TARGET_W - 350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

app = VideoDistanceApp()

def mouse_event_video(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if app.mode == "DISTANCE":
            app.target_point = (x, y)
            box_size = 40
            bbox = (max(0, x - box_size//2), max(0, y - box_size//2), box_size, box_size)
            try:
                app.target_tracker = cv2.TrackerCSRT_create()
                app.target_tracker.init(app.current_frame, bbox)
                print(f"[TARGET] Đã đặt & track điểm đích: {bbox}")
            except:
                print("[ERR] Không tạo được Target Tracker.")

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    app.init_calibration_maps((width, height))
    app.load_config()

    cv2.namedWindow("Smart ROI Tracking")
    cv2.setMouseCallback("Smart ROI Tracking", mouse_event_video)

    while True:
        if not app.paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Khi video loop lại, cần reset tracker nếu muốn chính xác tuyệt đối, 
                # nhưng ở đây ta cứ để nó chạy tiếp hoặc bạn có thể gọi init_roi_tracking lại.
                continue
            app.current_frame, target_box = app.process_frame(frame)
        else:
            target_box = None

        display_img = app.current_frame.copy()
        app.draw_overlays(display_img, target_box)
        cv2.imshow("Smart ROI Tracking", display_img)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'): break
        if key == ord(' '): app.paused = not app.paused
        if app.paused:
            if key == ord('h'): app.mode = "HEIGHT"
            if key == ord('d'): app.mode = "DISTANCE"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()