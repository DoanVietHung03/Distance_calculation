import cv2
import numpy as np
import math
import os
import json
import time
from ultralytics import YOLO
import torch

# IMPORT MODULE CŨ
from height_estimator import HeightEstimator

# ================= CẤU HÌNH =================
VIDEO_PATH = '..\\test_imgs\\cam_2\\cam_2.mp4' 
CALIB_FILE = '..\\calibration.json'
CONFIG_FILE = 'config.json'
TARGET_W = 1200 
YOLO_SKIP_FRAMES = 5

class VideoDistanceApp:
    def __init__(self):
        self.mode = "DISTANCE"
        self.paused = False
        self.frame_count = 0
        
        # --- STABILIZER CONFIG ---
        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        # Biến lưu trữ "Mỏ neo" (Anchor)
        self.gray_anchor = None        # Ảnh xám của Frame Gốc (lúc bắt đầu tracking)
        self.p0_anchor = None          # Các điểm đặc trưng ở Frame Gốc
        
        # --- ROI POINTS AND TARGET ---
        self.roi_points_initial = None # Tọa độ ROI gốc (cố định theo JSON)
        self.roi_points_curr = None    # Tọa độ ROI hiện tại (biến đổi theo M)
        # --- TARGET POINT ---
        self.target_point_initial = None
        self.target_point_curr = None
        
        # --- ANTI-OCCLUSION DATA ---
        self.last_known_boxes = [] # Lưu vị trí người để né
        
        self.target_tracker = None

        # Tools
        self.height_tool = HeightEstimator()
        self.yolo_model = None
        
        # Data
        self.real_world = {}
        self.cam_real_pos = (0.5, -18.0)
        self.clicked_points_orig = []
        self.matrix_homography = None
        self.scale_px_per_meter = 1.0 
        self.map1, self.map2 = None, None
        
        # Runtime
        self.current_frame = None
        self.detected_objects = []
        self.prev_time = 0
        self.fps = 0

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 0
            print(f"[INFO] GPU Activated: {torch.cuda.get_device_name(0)}")
        
        try:
            self.yolo_model = YOLO('..\\weights\\yolo11n-pose.onnx') 
        except: pass

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
        except: pass

    def load_config(self):
        if not os.path.exists(CONFIG_FILE): return False
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                self.real_world = data['real_world']
                self.cam_real_pos = (data['camera']['real_x'], data['camera']['real_y'])
                self.scale_px_per_meter = data['settings'].get('scale_px_per_meter', 1.0)
                
                # 1. LOAD ROI POINTS
                if 'points_px' in data:
                    self.clicked_points_orig = [tuple(p) for p in data['points_px']]
                    # Lưu bản gốc tuyệt đối
                    self.roi_points_initial = np.array(self.clicked_points_orig, dtype=np.float32).reshape(-1, 1, 2)
                    self.roi_points_curr = self.roi_points_initial.copy()
                    self.compute_homography(self.roi_points_curr)

                # 2. LOAD TARGET POINT
                if 'target_point' in data:
                    tp = data['target_point']
                    # Chuyển thành numpy array (1, 1, 2)
                    self.target_point_initial = np.array([[tp]], dtype=np.float32)
                    self.target_point_curr = self.target_point_initial.copy()
                    
                return True
        except: return False

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
        # Fix vector direction
        vx = (p3[0] - p1[0]) / d; vy = (p3[1] - p1[1]) / d
        rx = -vy; ry = vx
        p4 = (x0 + h * rx, y0 + h * ry)
        return [p1, p2, p3, p4]

    def compute_homography(self, current_pts_array):
        if len(current_pts_array) < 4: return
        rw = self.real_world
        real_coords = self.get_quadrilateral_coords(rw['L1'], rw['L2'], rw['L3'], rw['L4'], rw['diag_13'])
        if not real_coords: return
        dst_pts = np.float32([[pt[0]*self.scale_px_per_meter, pt[1]*self.scale_px_per_meter] for pt in real_coords])
        src_pts = current_pts_array.reshape(4, 2)
        try:
            self.matrix_homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
        except: pass

    def calculate_distance_points(self, p1, p2):
        if self.matrix_homography is None: return 0.0
        pts = np.float32([p1, p2]).reshape(-1, 1, 2)
        trans_pts = cv2.perspectiveTransform(pts, self.matrix_homography)
        dist_px = np.linalg.norm(trans_pts[0][0] - trans_pts[1][0])
        return dist_px / self.scale_px_per_meter

    # ================= ANCHOR STABILIZER (OPTIMIZED) =================
    
    # Đã xóa hàm is_point_in_boxes để tối ưu hóa

    def init_anchor(self, gray_frame):
        """
        Khởi tạo Frame Gốc (Anchor). Mọi frame sau này sẽ được so sánh trực tiếp với Frame này.
        """
        self.gray_anchor = gray_frame.copy()
        
        # Mask 1: Loại bỏ vùng bên ngoài ROI (nếu có)
        mask = np.ones_like(gray_frame, dtype=np.uint8) * 255
        if self.roi_points_initial is not None:
            pts = self.roi_points_initial.astype(np.int32)
            cv2.fillPoly(mask, [pts], 0) 

        # Mask 2: Loại bỏ vùng đang có người ngay từ đầu
        if len(self.last_known_boxes) > 0:
            for box in self.last_known_boxes:
                x1, y1, x2, y2 = box
                # Vẽ màu đen (0) vào vùng có người để goodFeaturesToTrack bỏ qua
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        
        self.p0_anchor = cv2.goodFeaturesToTrack(self.gray_anchor, mask=mask, maxCorners=300, qualityLevel=0.01, minDistance=10)
        print(f"[STABILIZER] Anchor Reset. Points: {len(self.p0_anchor) if self.p0_anchor is not None else 0}")

    def update_stabilizer(self, gray_curr):
        """
        Tính toán M: Frame Gốc -> Frame Hiện tại
        """
        if self.p0_anchor is None or len(self.p0_anchor) < 10:
            self.init_anchor(gray_curr)
            return

        # Tính Optical Flow từ ANCHOR -> CURRENT
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.gray_anchor, gray_curr, self.p0_anchor, None, **self.lk_params)
        
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.p0_anchor[st == 1]
            
            # --- TỐI ƯU HÓA: Dùng Mask Numpy thay vì vòng lặp for ---
            # Chỉ giữ lại các điểm KHÔNG nằm đè lên người
            if len(self.last_known_boxes) > 0:
                h_img, w_img = gray_curr.shape
                # Tạo mask đen (0)
                mask_boxes = np.zeros((h_img, w_img), dtype=np.uint8)
                
                # Vẽ các box màu trắng (255) lên mask
                for box in self.last_known_boxes:
                    x1, y1, x2, y2 = box
                    # Mở rộng pad 10px như logic cũ
                    cv2.rectangle(mask_boxes, (max(0, x1-10), max(0, y1-10)), (min(w_img, x2+10), min(h_img, y2+10)), 255, -1)
                
                # Chuyển tọa độ điểm thành số nguyên để index matrix
                pts_int = good_new.astype(int)
                
                # Clip tọa độ để không bị lỗi index out of bounds
                pts_int[:, 0] = np.clip(pts_int[:, 0], 0, w_img - 1)
                pts_int[:, 1] = np.clip(pts_int[:, 1], 0, h_img - 1)
                
                # Kiểm tra điểm nào rơi vào vùng màu trắng (255)
                # mask_boxes[y, x] > 0
                is_on_person = mask_boxes[pts_int[:, 1], pts_int[:, 0]] > 0
                
                # Giữ lại điểm KHÔNG nằm trên người (~is_on_person)
                clean_new = good_new[~is_on_person]
                clean_old = good_old[~is_on_person]
            else:
                clean_new = good_new
                clean_old = good_old
            # ------------------------------------------------

            if len(clean_new) > 10: # Cần đủ điểm sạch để tính
                M, mask_ransac = cv2.findHomography(clean_old, clean_new, cv2.RANSAC, 5.0)
                
                if M is not None:
                    # Biến đổi ROI Gốc theo M để ra ROI Hiện tại
                    # Luôn transform từ ROI GỐC (initial), không lấy cái cũ transform tiếp
                    if self.roi_points_initial is not None:
                        self.roi_points_curr = cv2.perspectiveTransform(self.roi_points_initial, M)
                        self.compute_homography(self.roi_points_curr)
                    
                    # Transform TARGET POINT 
                    if self.target_point_initial is not None:
                        self.target_point_curr = cv2.perspectiveTransform(self.target_point_initial, M)

            # Reset nếu mất quá nhiều điểm
            track_ratio = len(clean_new) / (len(self.p0_anchor) + 1e-5) # +1e-5 để tránh chia cho 0
            if track_ratio < 0.3:
                print("[STABILIZER] Điểm sạch quá ít -> Reset Anchor.")
                self.init_anchor(gray_curr)

    def get_current_target_tuple(self):
        """Helper để lấy tọa độ (x, y) từ numpy array"""
        if self.target_point_curr is not None:
            return tuple(self.target_point_curr[0][0].astype(int))
        return None

    def update_target_tracker(self, frame):
        # Giữ lại logic cũ (mouse click tracker) phòng khi cần
        track_box = None
        if self.target_tracker is not None:
            success, box = self.target_tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                # Nếu Tracker (KCF) đang chạy, nó sẽ override Stabilizer cho target
                pt = (x + w//2, y + h//2)
                self.target_point_curr = np.array([[pt]], dtype=np.float32)
                track_box = (x, y, w, h)
            else:
                self.target_tracker = None
        return track_box

    def detect_objects_and_update_boxes(self, img):
        """Hàm này vừa detect YOLO vừa cập nhật box cho Stabilizer né"""
        if self.yolo_model is None: return

        # Chỉ chạy YOLO mỗi N frames để tối ưu, nhưng vẫn giữ box cũ
        if self.frame_count % YOLO_SKIP_FRAMES == 0:
            results = self.yolo_model(img, verbose=False, device=self.device, conf=0.5, imgsz=640)
            self.detected_objects = []
            self.last_known_boxes = [] # Reset box cũ
            
            target_pt = self.get_current_target_tuple()

            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                kpts_data = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
                
                for i, box in enumerate(boxes):
                    # Lưu box vào danh sách "cấm" cho Stabilizer
                    self.last_known_boxes.append(box)
                    
                    x1, y1, x2, y2 = box
                    ground_point = (int((x1+x2)/2), y2)
                    head_point = (int((x1+x2)/2), y1)
                    
                    if kpts_data is not None and len(kpts_data) > i:
                        kp = kpts_data[i]
                        if kp[15][2] > 0.5 and kp[16][2] > 0.5:
                            ground_point = (int((kp[15][0]+kp[16][0])/2), int((kp[15][1]+kp[16][1]+5)/2))
                        if kp[0][2] > 0.5:
                            head_point = (int(kp[0][0]), int(kp[0][1]))

                    obj_info = {'box': box, 'head': head_point, 'foot': ground_point, 'h_real': 0.0, 'd_to_target': 0.0}
                    
                    if self.mode == "HEIGHT":
                        h_real, _ = self.height_tool.calculate(head_point, ground_point, self.matrix_homography, self.cam_real_pos)
                        obj_info['h_real'] = h_real
                    elif self.mode == "DISTANCE" and target_pt:
                        d_target = self.calculate_distance_points(ground_point, target_pt)
                        obj_info['d_to_target'] = d_target
                    
                    self.detected_objects.append(obj_info)
        else:
            # Nếu skip frame YOLO, ta vẫn cần tính lại khoảng cách cho các object cũ
            # (Vì camera có thể rung, object vẫn di chuyển trong logic code cũ, 
            # nhưng ở đây ta chỉ dùng box cũ để né Optical Flow thôi)
            pass

    def process_frame(self, raw_frame):
        curr_time = time.time()
        self.fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = curr_time

        if self.map1 is not None:
            frame = cv2.remap(raw_frame, self.map1, self.map2, cv2.INTER_LINEAR)
        else: frame = raw_frame
        
        h, w = frame.shape[:2]
        scale = TARGET_W / w
        new_h = int(h * scale)
        frame_resized = cv2.resize(frame, (TARGET_W, new_h))
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # 1. Chạy Detect trước để lấy Bounding Box (cho Stabilizer né)
        # Lưu ý: Logic cũ chạy detect sau, nhưng giờ ta cần vị trí người TRƯỚC khi tính flow
        # Tuy nhiên, nếu chạy detect trước thì hơi lag nếu máy yếu. 
        # Tốt nhất: Dùng Box của frame TRƯỚC ĐÓ (self.last_known_boxes) để né cho frame NAY.
        # Sai số vị trí người giữa 1 frame (33ms) là không đáng kể.
        
        # 2. Update Stabilizer (Dùng box của frame cũ hoặc frame detect gần nhất)
        if self.gray_anchor is None:
            self.init_anchor(frame_gray)
        else:
            self.update_stabilizer(frame_gray)
            
        target_box = self.update_target_tracker(frame_resized)
        
        # 3. Detect object (Cập nhật box mới cho vòng lặp sau)
        self.detect_objects_and_update_boxes(frame_resized)
        
        self.frame_count += 1
        return frame_resized, target_box

    def draw_overlays(self, img, target_box):
        cv2.putText(img, f"FPS: {int(self.fps)}", (TARGET_W - 120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ROI - Vẽ màu đỏ
        if self.roi_points_curr is not None:
            pts = self.roi_points_curr.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)

        target_pt = self.get_current_target_tuple()

        if self.mode == "DISTANCE" and target_pt:
            cv2.circle(img, target_pt, 5, (0, 255, 255), -1)
            cv2.circle(img, target_pt, 12, (0, 255, 255), 2)
            if target_box:
                tx, ty, tw, th = target_box
                cv2.rectangle(img, (tx, ty), (tx+tw, ty+th), (0, 255, 255), 1)

        # Objects
        for obj in self.detected_objects:
            x1, y1, x2, y2 = obj['box']
            foot = obj['foot']
            cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), 1)

            if self.mode == "HEIGHT":
                cv2.line(img, obj['head'], foot, (0, 255, 0), 2)
                cv2.putText(img, f"H: {obj['h_real']:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif self.mode == "DISTANCE":
                cv2.circle(img, foot, 4, (255, 0, 0), -1)
                if target_pt:
                    cv2.line(img, foot, target_pt, (0, 165, 255), 2)
                    mid = ((foot[0]+target_pt[0])//2, (foot[1]+target_pt[1])//2)
                    cv2.putText(img, f"{obj['d_to_target']:.2f}m", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # UI
        status_text = "PAUSED" if self.paused else "PLAYING"
        cv2.rectangle(img, (0, 0), (TARGET_W, 60), (0, 0, 0), -1)
        cv2.putText(img, f"MODE: {self.mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, status_text, (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if self.paused else (0, 255, 0), 1)

app = VideoDistanceApp()

def mouse_event_video(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if app.mode == "DISTANCE":
            # Nếu click chuột, ta cập nhật thủ công vào biến curr
            app.target_point_curr = np.array([[[x, y]]], dtype=np.float32)
            
            # Cố gắng reset tracker KCF để track theo điểm mới (dù Anchor Stabilizer vẫn đang chạy ngầm)
            box_size = 20
            bbox = (max(0, x - box_size//2), max(0, y - box_size//2), box_size, box_size)
            try:
                app.target_tracker = cv2.TrackerKCF_create() 
                app.target_tracker.init(app.current_frame, bbox)
                print(f"[TARGET] Mouse Click override: {bbox}")
            except: pass

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    app.init_calibration_maps((width, height))
    app.load_config()

    cv2.namedWindow("Anchor Tracking")
    cv2.setMouseCallback("Anchor Tracking", mouse_event_video)

    while True:
        if not app.paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Khi loop video, phải reset Anchor vì cảnh quay lại từ đầu
                app.gray_anchor = None 
                continue
            app.current_frame, target_box = app.process_frame(frame)
        else:
            target_box = None

        display_img = app.current_frame.copy()
        app.draw_overlays(display_img, target_box)
        cv2.imshow("Anchor Tracking", display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord(' '): app.paused = not app.paused
        if app.paused:
            if key == ord('h'): app.mode = "HEIGHT"
            if key == ord('d'): app.mode = "DISTANCE"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
    
# 1. Điểm nghẽn nghiêm trọng nhất: Thứ tự Undistort và Resize
# Hiện tại: Bạn đang gọi cv2.remap (khử méo) trên frame gốc (Full HD hoặc 4K) trước khi resize về 1200px.

# Vấn đề: cv2.remap là một phép toán cực nặng (tính toán lại từng pixel). Làm việc này trên ảnh 4K tốn gấp 4-8 lần so với ảnh 1200px.

# Giải pháp: Resize ảnh gốc trước -> Sau đó mới Remap. Bạn cần scale lại Ma trận Camera (K) tương ứng với tỷ lệ resize.