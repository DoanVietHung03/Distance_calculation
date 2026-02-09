import cv2
import numpy as np
import math
import csv
import os
import json
import time
from ultralytics import YOLO
import torch

# IMPORT MODULE MỚI
from height_estimator import HeightEstimator

# ================= CẤU HÌNH BAN ĐẦU =================
IMAGE_PATH = '..\\test_imgs\\cam_2\\cam_2.jpg'       
CALIB_FILE = '..\\calibration.json' 
CONFIG_FILE = 'config.json' # File cấu hình mới
CSV_FILE_NAME = 'measurement_data.csv'

class DistanceApp:
    def __init__(self):
        # --- CẤU HÌNH: CHẾ ĐỘ ĐO ---
        self.mode = "DISTANCE" 
        self.height_clicks = [] 
        
        # Khởi tạo công cụ tính chiều cao
        self.height_tool = HeightEstimator()

        # --- CẤU HÌNH CHẾ ĐỘ ---
        self.clicked_points = []     
        self.measure_points = []     
        self.matrix_homography = None
        self.scale_px_per_meter = 1.0 
        
        # Biến lưu trữ thông số thực tế (Load từ JSON)
        self.real_world = {} 
        self.cam_real_pos = (0.5, -18.0) # Default fallback

        self.drawing = False
        self.ix, self.iy = -1, -1
        self.cur_mouse = (-1, -1) 
        
        self.clean_frame = None   
        self.orig_resized = None  
        self.last_click_time = 0

        self.yolo_model = None
        self.detected_objects = []
        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 0
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n[INFO] Đã kích hoạt GPU: {gpu_name}")
        else:
            print("\n[WARN] Không tìm thấy GPU NVIDIA. Đang chạy bằng CPU.")
            
        try:
            self.yolo_model = YOLO('..\\weights\\yolo11n-pose.onnx') 
            print("Đã load model YOLO11n...")
        except:
            print("[ERR] Không tìm thấy model weights!")

    def load_config(self, config_path):
        """
        Đọc file config.json và setup các thông số ban đầu
        """
        if not os.path.exists(config_path):
            print(f"[ERR] Không tìm thấy file {config_path}")
            return False

        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            # 1. Load thông số kích thước thực
            self.real_world = data['real_world']
            print(f"[CONFIG] Real World Loaded: {self.real_world}")

            # 2. Load thông số Camera
            cam = data['camera']
            self.cam_real_pos = (cam['real_x'], cam['real_y'])
            print(f"[CONFIG] Camera Position: {self.cam_real_pos}")

            # 3. Load Settings
            if 'settings' in data:
                self.scale_px_per_meter = data['settings'].get('scale_px_per_meter', 1.0)
                print(f"[CONFIG] Scale (px/m): {self.scale_px_per_meter}") 

            # 4. Load 4 điểm Pixel đã chọn trước (Quan trọng)
            if 'points_px' in data and len(data['points_px']) == 4:
                self.clicked_points = [tuple(p) for p in data['points_px']]
                print(f"[CONFIG] Loaded 4 points: {self.clicked_points}")
                return True
            else:
                print("[ERR] Config thiếu 'points_px' hoặc không đủ 4 điểm.")
                return False

        except Exception as e:
            print(f"[ERR] Lỗi đọc config: {e}")
            return False

    # --- CÁC HÀM HỖ TRỢ KHÁC ---
    def get_quadrilateral_coords(self, l1, l2, l3, l4, d13):
        if l1 + l2 < d13 or abs(l1 - l2) > d13:
            print("[ERR] Số liệu sai hình học!")
            return []
        p1 = (0.0, 0.0)
        p2 = (l1, 0.0)
        cos_alpha = (l1**2 + d13**2 - l2**2) / (2 * l1 * d13)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        alpha = math.acos(cos_alpha)
        p3 = (d13 * math.cos(alpha), d13 * math.sin(alpha))
        d = d13
        a = (l4**2 - l3**2 + d**2) / (2*d)
        val_sqrt = l4**2 - a**2
        h = math.sqrt(max(0, val_sqrt))
        x0 = p1[0] + a * (p3[0] - p1[0]) / d
        y0 = p1[1] + a * (p3[1] - p1[1]) / d
        rx = -(p3[1] - p1[1]) / d
        ry = (p3[0] - p1[0]) / d
        p4_a = (x0 + h * rx, y0 + h * ry)
        p4_b = (x0 - h * rx, y0 - h * ry)
        def cross_product(vx, vy, px, py): return vx * py - vy * px
        if cross_product(p3[0], p3[1], p4_a[0], p4_a[1]) > 0: p4 = p4_a
        else: p4 = p4_b
        return [p1, p2, p3, p4]

    def load_calibration_and_undistort(self, img_path):
        img = cv2.imread(img_path)
        if img is None: return None
        if not os.path.exists(CALIB_FILE): return img
        try:
            with open(CALIB_FILE, 'r') as f: data = json.load(f)
            print(f"[OK] Đã load file Calibration: {CALIB_FILE}")
            K = np.array(data['camera_matrix'])
            D = np.array(data['distortion_coefficients'])
            if 'image_resolution' in data: calib_w, calib_h = data['image_resolution']
            else:
                print("[ERR] Không tìm thấy thống tin kích thước hình anh.") 
                calib_w, calib_h = 810, 720
            h_curr, w_curr = img.shape[:2]
            if w_curr != calib_w or h_curr != calib_h:
                scale_x = w_curr / calib_w; scale_y = h_curr / calib_h
                K[0, 0] *= scale_x; K[1, 1] *= scale_y; K[0, 2] *= scale_x; K[1, 2] *= scale_y
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w_curr, h_curr), 1, (w_curr, h_curr))
            map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, (w_curr, h_curr), 5)
            undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            return undistorted
        except Exception as e:
            print(f"[ERR] Lỗi Calibration: {e}")
            return img

    def get_bbox_ground_point(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        ground_x = int((x_min + x_max) / 2)
        ground_y = int(y_max)
        return (ground_x, ground_y), (x_min, y_min, x_max, y_max)
    
    def check_valid_convex(self, points):
        if len(points) != 4: return False
        pts_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        return cv2.isContourConvex(pts_array)

    def compute_homography(self):
        if len(self.clicked_points) < 4: return
        
        # Lấy thông số từ self.real_world đã load
        rw = self.real_world
        if not rw: return 

        real_coords = self.get_quadrilateral_coords(
            rw['L1'], rw['L2'], rw['L3'], rw['L4'], rw['diag_13']
        )
        
        if not real_coords: return
        dst_pts = np.float32([[pt[0] * self.scale_px_per_meter, pt[1] * self.scale_px_per_meter] for pt in real_coords])
        src_pts = np.float32(self.clicked_points)
        self.matrix_homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
        print(f"[OK] Homography Calculated Automatically.")

    def calculate_distance_real(self, p1, p2):
        if self.matrix_homography is None: return 0.0
        pts = np.float32([p1, p2]).reshape(-1, 1, 2)
        trans_pts = cv2.perspectiveTransform(pts, self.matrix_homography)
        dist_px = np.linalg.norm(trans_pts[0][0] - trans_pts[1][0])
        return dist_px / self.scale_px_per_meter

    def save_csv(self, p_start, p_end, dist_m):
        file_exists = os.path.isfile(CSV_FILE_NAME)
        with open(CSV_FILE_NAME, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Ref_Pixels', 'P_Start', 'P_End', 'Distance_Meter'])
            writer.writerow([str(self.clicked_points), str(p_start), str(p_end), round(dist_m, 4)])
            print(f"[CSV] Saved: {dist_m:.2f}m")

    def detect_objects(self):
        if self.yolo_model is None or self.orig_resized is None: return
        print("Đang chạy YOLO POSE detect...")
        results = self.yolo_model(self.orig_resized, verbose=False, device=self.device)
        self.detected_objects = []
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            if r.keypoints is not None and r.keypoints.data is not None:
                all_keypoints = r.keypoints.data.cpu().numpy()
            else:
                all_keypoints = None

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                w_box = x2 - x1
                h_box = y2 - y1
                
                ground_point = None
                head_point = None 
                ankles = []
                method = "BBOX"
                IS_FAR_AWAY = h_box < 60

                if not IS_FAR_AWAY and all_keypoints is not None and len(all_keypoints) > i:
                    kpts = all_keypoints[i]
                    left_ankle = kpts[15]
                    right_ankle = kpts[16]
                    l_ok = left_ankle[2] > 0.5
                    r_ok = right_ankle[2] > 0.5
                    
                    if l_ok and r_ok:
                        gx = int((left_ankle[0] + right_ankle[0]) / 2)
                        gy = int((left_ankle[1] + right_ankle[1]) / 2)
                        ground_point = (gx, gy)
                        ankles = [(int(left_ankle[0]), int(left_ankle[1])), 
                                  (int(right_ankle[0]), int(right_ankle[1]))]
                        method = "POSE"
                    elif l_ok:
                        ground_point = (int(left_ankle[0]), int(left_ankle[1]))
                        ankles = [ground_point]
                        method = "POSE"
                    elif r_ok:
                        ground_point = (int(right_ankle[0]), int(right_ankle[1]))
                        ankles = [ground_point]
                        method = "POSE"

                    nose = kpts[0]
                    if nose[2] > 0.3:
                        head_point = (int(nose[0]), int(nose[1]))
                    else:
                        head_point = (int((x1 + x2) / 2), y1)

                if ground_point is None:
                    ground_point = (int((x1 + x2) / 2), int(y2))
                    method = "BBOX_FAR" if IS_FAR_AWAY else "BBOX_FALLBACK"
                
                if head_point is None:
                    head_point = (int((x1 + x2) / 2), y1)

                self.detected_objects.append({
                    'box': (x1, y1, x2, y2),
                    'ground_point': ground_point,
                    'head_point': head_point, 
                    'ankles': ankles,
                    'method': method
                })
        print(f"-> Tìm thấy {len(self.detected_objects)} người.")

# ================= MAIN LOOP & MOUSE EVENTS =================
app = DistanceApp()

def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        app.cur_mouse = (x, y)

    # Đã bỏ qua phần Setup click 4 điểm vì load từ config
    if app.matrix_homography is not None:
        # --- CHẾ ĐỘ ĐO ĐẠC ---
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_person = None
            for obj in app.detected_objects:
                bx1, by1, bx2, by2 = obj['box']
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    clicked_person = obj
                    break

            if app.mode == "DISTANCE":
                if not app.drawing: 
                    app.drawing = True
                    app.ix, app.iy = x, y
                
            elif app.mode == "HEIGHT":
                if clicked_person:
                    # AUTO
                    head = clicked_person['head_point']
                    foot = clicked_person['ground_point']
                    # Sử dụng app.cam_real_pos load từ JSON
                    h_real, d_real = app.height_tool.calculate(head, foot, app.matrix_homography, app.cam_real_pos)
                    
                    cv2.line(app.clean_frame, head, foot, (0, 255, 0), 2)
                    cv2.circle(app.clean_frame, head, 4, (0, 0, 255), -1)
                    cv2.circle(app.clean_frame, foot, 4, (0, 255, 255), -1)
                    
                    label = f"H: {h_real:.2f}m (D: {d_real:.1f}m)"
                    cv2.putText(app.clean_frame, label, (head[0], head[1]-15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    print(f"[HEIGHT] {label}")
                
                else:
                    # MANUAL
                    app.height_clicks.append((x,y))
                    cv2.circle(app.clean_frame, (x,y), 4, (0, 255, 0), -1)
                    
                    if len(app.height_clicks) == 2:
                        p1, p2 = app.height_clicks[-2], app.height_clicks[-1]
                        if p1[1] > p2[1]: foot, head = p1, p2
                        else: foot, head = p2, p1
                        
                        # Sử dụng app.cam_real_pos load từ JSON
                        h_real, d_real = app.height_tool.calculate(head, foot, app.matrix_homography, app.cam_real_pos)
                        
                        cv2.line(app.clean_frame, head, foot, (0, 255, 0), 2)
                        label = f"H: {h_real:.2f}m (D from cam: {d_real:.1f}m)"
                        cv2.putText(app.clean_frame, label, head, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        print(f"[HEIGHT MANUAL] {label}")
                        app.height_clicks = [] 

        elif event == cv2.EVENT_LBUTTONUP:
            if app.mode == "DISTANCE" and app.drawing:
                app.drawing = False
                drag_dist = math.hypot(x - app.ix, y - app.iy)
                final_point = None
                
                if drag_dist > 10: 
                    ground_pt, bbox = app.get_bbox_ground_point((app.ix, app.iy), (x, y))
                    final_point = ground_pt
                    cv2.rectangle(app.clean_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.circle(app.clean_frame, ground_pt, 5, (0, 255, 255), -1)
                else: 
                    clicked_person = None
                    for obj in app.detected_objects:
                        bx1, by1, bx2, by2 = obj['box']
                        if bx1 <= x <= bx2 and by1 <= y <= by2:
                            clicked_person = obj
                            break
                    
                    if clicked_person:
                        final_point = clicked_person['ground_point']
                        for ank in clicked_person['ankles']:
                             cv2.circle(app.clean_frame, ank, 4, (0, 0, 255), -1)
                        cv2.circle(app.clean_frame, final_point, 6, (0, 255, 255), -1)
                    else:
                        final_point = (x, y)
                        cv2.circle(app.clean_frame, final_point, 5, (255, 0, 255), -1)

                if final_point:
                    app.measure_points.append(final_point)
                    if len(app.measure_points) >= 2 and len(app.measure_points) % 2 == 0:
                        p_start = app.measure_points[-2]
                        p_end = app.measure_points[-1]
                        dist = app.calculate_distance_real(p_start, p_end)
                        cv2.line(app.clean_frame, p_start, p_end, (0, 165, 255), 2)
                        mid = ((p_start[0]+p_end[0])//2, (p_start[1]+p_end[1])//2)
                        cv2.putText(app.clean_frame, f"{dist:.2f}m", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                        app.save_csv(p_start, p_end, dist)

def main():
    img_undistorted = app.load_calibration_and_undistort(IMAGE_PATH)
    if img_undistorted is None: return

    h_orig, w_orig = img_undistorted.shape[:2]
    TARGET_W = 1200 
    scale = TARGET_W / w_orig
    new_h = int(h_orig * scale)
    
    app.orig_resized = cv2.resize(img_undistorted, (TARGET_W, new_h))
    app.clean_frame = app.orig_resized.copy()

    # --- SETUP HEIGHT ESTIMATOR ---
    app.height_tool.load_focal_length(CALIB_FILE, TARGET_W)

    # --- AUTO CONFIG TỪ JSON (MỚI) ---
    if app.load_config(CONFIG_FILE):
        # 1. Tính toán Homography luôn
        app.compute_homography()
        
        # 2. Vẽ sẵn khung 4 điểm lên Clean Frame để user thấy vùng đã chọn
        pts = np.array(app.clicked_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(app.clean_frame, [pts], True, (0, 0, 255), 2)
        for i, p in enumerate(app.clicked_points):
            cv2.circle(app.clean_frame, p, 5, (0, 255, 255), -1)
            cv2.putText(app.clean_frame, f"{i+1}", (p[0]+10, p[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        print("[WARN] Không load được config. Bạn có thể cần code lại phần Setup thủ công nếu muốn.")

    app.detect_objects()
    
    print(f"Ảnh làm việc: {TARGET_W}x{new_h}")
    print("\n--- HƯỚNG DẪN SỬ DỤNG ---")
    print("Hệ thống đã tự động setup vùng đo từ config.json")
    print("1. Phím 'd': Chế độ ĐO KHOẢNG CÁCH SÀN.")
    print("2. Phím 'h': Chế độ ĐO CHIỀU CAO.")
    print("   - Auto: Click vào người.")
    print("   - Manual: Click Chân -> Click Đầu.")
    print("3. 'r': Reset, 'q': Thoát.")

    cv2.namedWindow("Smart Distance")
    cv2.setMouseCallback("Smart Distance", mouse_event)

    while True:
        img_show = app.clean_frame.copy()

        # Hiển thị chế độ hiện tại
        mode_color = (0, 165, 255) if app.mode == "DISTANCE" else (0, 255, 0)
        cv2.putText(img_show, f"MODE: {app.mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, mode_color, 2)

        if not app.drawing:
            for obj in app.detected_objects:
                bx1, by1, bx2, by2 = obj['box']
                gx, gy = obj['ground_point']
                
                # Vẽ box xám mờ
                cv2.rectangle(img_show, (bx1, by1), (bx2, by2), (100, 100, 100), 1)
                
                if app.mode == "DISTANCE":
                    # Gợi ý điểm chân
                    cv2.circle(img_show, (gx, gy), 4, (255, 200, 0), -1) 
                
                elif app.mode == "HEIGHT":
                    # Gợi ý đường cao (Đầu -> Chân)
                    head = obj['head_point']
                    cv2.line(img_show, head, (gx, gy), (100, 100, 100), 1)
                    cv2.circle(img_show, head, 3, (0, 255, 0), -1)

        # Draw Zoom window & Preview
        if app.cur_mouse != (-1, -1):
            mx, my = app.cur_mouse
            zoom_factor = 3      
            crop_sz = 40         
            x1 = max(0, mx - crop_sz); y1 = max(0, my - crop_sz)
            x2 = min(TARGET_W, mx + crop_sz); y2 = min(new_h, my + crop_sz)
            roi = app.clean_frame[y1:y2, x1:x2]
            if roi.size > 0:
                zoomed = cv2.resize(roi, (0,0), fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_NEAREST)
                zh, zw = zoomed.shape[:2]
                cv2.rectangle(zoomed, (0,0), (zw-1, zh-1), (255, 255, 255), 2)
                cv2.line(zoomed, (zw//2, 0), (zw//2, zh), (0,0,255), 1)
                cv2.line(zoomed, (0, zh//2), (zw, zh//2), (0,0,255), 1)
                margin = 20
                if zw < TARGET_W and zh < new_h:
                    img_show[margin:margin+zh, TARGET_W-margin-zw:TARGET_W-margin] = zoomed

        cv2.imshow("Smart Distance", img_show)
        key = cv2.waitKey(1)
        if key == ord('q'): break
        
        # --- LOGIC CHUYỂN CHẾ ĐỘ ---
        if key == ord('d'): app.mode = "DISTANCE"
        if key == ord('h'): 
            app.mode = "HEIGHT"
            app.height_clicks = []
            
        if key == ord('r'):
            # Reset cơ bản, nhưng vẫn giữ homography đã load từ json
            app.measure_points = []
            app.clean_frame = app.orig_resized.copy()
            # Vẽ lại polygon vùng đo sau khi reset
            if len(app.clicked_points) == 4:
                pts = np.array(app.clicked_points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(app.clean_frame, [pts], True, (0, 0, 255), 2)
                for i, p in enumerate(app.clicked_points):
                    cv2.circle(app.clean_frame, p, 5, (0, 255, 255), -1)
                    cv2.putText(app.clean_frame, f"{i+1}", (p[0]+10, p[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            app.detect_objects()
            print("--- RESET DRAWING ---")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()