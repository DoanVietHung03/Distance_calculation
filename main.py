import cv2
import numpy as np
import math
import csv
import os
import json
import time

# ================= CẤU HÌNH HÌNH HỌC SÀN NHÀ =================
IMAGE_PATH = '.\\test_imgs\\test_5.jpg'       
CALIB_FILE = 'calibration.json' 
CSV_FILE_NAME = 'measurement_data.csv'

# Kích thước thực tế (Mét)
L1 = 3.45       # Top
L2 = 10.74      # Right
L3 = 3.2        # Bottom
L4 = 12.0       # Left
DIAG_13 = 11.54 # Diagonal
# ============================================================

class DistanceApp:
    def __init__(self):
        self.clicked_points = []     
        self.measure_points = []     
        self.matrix_homography = None
        self.scale_px_per_meter = 100 
        
        # Trạng thái chuột
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.cur_mouse = (-1, -1) 
        
        self.clean_frame = None   
        self.orig_resized = None  
        self.last_click_time = 0

    def get_quadrilateral_coords(self, l1, l2, l3, l4, d13):
        """Thuật toán tái tạo hình học sàn nhà"""
        if l1 + l2 < d13 or abs(l1 - l2) > d13:
            print("[ERR] Số liệu sai: L1, L2 và Đường chéo không tạo thành tam giác!")
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
            K = np.array(data['camera_matrix'])
            D = np.array(data['distortion_coefficients'])
            if 'image_resolution' in data: calib_w, calib_h = data['image_resolution']
            else: calib_w, calib_h = 810, 720
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
        """Tính điểm chân từ BBox"""
        x1, y1 = p1
        x2, y2 = p2
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        ground_x = int((x_min + x_max) / 2)
        ground_y = int(y_max)
        return (ground_x, ground_y), (x_min, y_min, x_max, y_max)
    
    # Kiểm tra tứ giác lồi
    def check_valid_convex(self, points):
        if len(points) != 4: return False
        # OpenCV yêu cầu định dạng array shape (N, 1, 2)
        pts_array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        return cv2.isContourConvex(pts_array)

    def compute_homography(self):
        if len(self.clicked_points) < 4: return
        real_coords = self.get_quadrilateral_coords(L1, L2, L3, L4, DIAG_13)
        if not real_coords: return

        dst_pts = np.float32([[pt[0] * self.scale_px_per_meter, pt[1] * self.scale_px_per_meter] for pt in real_coords])
        src_pts = np.float32(self.clicked_points)
        self.matrix_homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
        print(f"[OK] Đã tính Homography.")

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

# ================= MAIN LOOP & MOUSE EVENTS =================
app = DistanceApp()

def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        app.cur_mouse = (x, y)

    if app.matrix_homography is None:
        # --- CHẾ ĐỘ SETUP ---
        if event == cv2.EVENT_LBUTTONDOWN:
            if time.time() - app.last_click_time < 0.3: return 
            app.last_click_time = time.time()
            
            if len(app.clicked_points) < 4:
                app.clicked_points.append((x, y))
                cv2.circle(app.clean_frame, (x, y), 5, (0, 0, 255), -1)
                
                if len(app.clicked_points) > 1:
                    cv2.line(app.clean_frame, app.clicked_points[-2], (x,y), (0,0,255), 1)
                
                # Khi đủ 4 điểm: Kiểm tra và tính toán
                if len(app.clicked_points) == 4:
                    cv2.line(app.clean_frame, app.clicked_points[3], app.clicked_points[0], (0,0,255), 1)
                    
                    # Validate Tứ Giác
                    if app.check_valid_convex(app.clicked_points):
                        app.compute_homography()
                        print("\n>>> SETUP XONG. CHẾ ĐỘ ĐO KÍCH HOẠT <<<")
                    else:
                        print("\n[CẢNH BÁO] 4 điểm không tạo thành tứ giác lồi (bị chéo hoặc lõm)!")
                        print(">> Vui lòng Reset (nhấn 'r') và chọn lại theo thứ tự vòng tròn.")
                        # Xóa kết nối điểm cuối để người dùng nhận ra lỗi
                        app.clicked_points = []
                        app.clean_frame = app.orig_resized.copy() # Reset visual
    else:
        # --- CHẾ ĐỘ ĐO ---
        if event == cv2.EVENT_LBUTTONDOWN:
            app.drawing = True
            app.ix, app.iy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            if app.drawing: 
                app.drawing = False
                drag_dist = math.hypot(x - app.ix, y - app.iy)
                final_point = None
                
                # Kéo > 10px -> BBox
                if drag_dist > 10:
                    ground_pt, bbox = app.get_bbox_ground_point((app.ix, app.iy), (x, y))
                    final_point = ground_pt
                    cv2.rectangle(app.clean_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.circle(app.clean_frame, ground_pt, 5, (0, 255, 255), -1)
                    print(f"-> Đã chọn BBox tại {ground_pt}")
                # Click -> Point
                else:
                    final_point = (x, y)
                    cv2.circle(app.clean_frame, final_point, 5, (255, 0, 255), -1)
                    cv2.circle(app.clean_frame, final_point, 9, (255, 0, 255), 1)
                    print(f"-> Đã chấm điểm Point tại {final_point}")

                if final_point:
                    app.measure_points.append(final_point)
                    if len(app.measure_points) >= 2 and len(app.measure_points) % 2 == 0:
                        p_start = app.measure_points[-2]
                        p_end = app.measure_points[-1]
                        dist = app.calculate_distance_real(p_start, p_end)
                        
                        cv2.line(app.clean_frame, p_start, p_end, (0, 165, 255), 2)
                        mid_x = (p_start[0] + p_end[0]) // 2
                        mid_y = (p_start[1] + p_end[1]) // 2
                        cv2.putText(app.clean_frame, f"{dist:.2f}m", (mid_x, mid_y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
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
    
    print(f"Ảnh làm việc: {TARGET_W}x{new_h}")
    print("\n--- HƯỚNG DẪN SỬ DỤNG ---")
    print("1. SETUP: Click 4 điểm góc sàn.")
    print("2. ĐO KHOẢNG CÁCH (Kéo BBox hoặc Click điểm).")
    print("3. Phím 'r': Reset. Phím 'q': Thoát.")

    cv2.namedWindow("Smart Distance")
    cv2.setMouseCallback("Smart Distance", mouse_event)

    while True:
        img_show = app.clean_frame.copy()

        # 1. Vẽ BBox Preview (nếu đang kéo chuột)
        if app.drawing and app.cur_mouse != (-1, -1):
            drag_dist_preview = math.hypot(app.cur_mouse[0] - app.ix, app.cur_mouse[1] - app.iy)
            if drag_dist_preview > 10:
                cv2.rectangle(img_show, (app.ix, app.iy), app.cur_mouse, (0, 255, 0), 2)
            else:
                cv2.drawMarker(img_show, app.cur_mouse, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10)
        
        # 2. VẼ KÍNH LÚP (ZOOM WINDOW) KHI SETUP
        # Chỉ hiện khi chưa tính Homography và chuột đang trong cửa sổ
        if app.matrix_homography is None and app.cur_mouse != (-1, -1):
            mx, my = app.cur_mouse
            zoom_factor = 4      # Phóng to 4 lần
            crop_sz = 40         # Vùng cắt quanh chuột (40x40 px)
            
            # Giới hạn vùng cắt không vượt quá ảnh
            x1 = max(0, mx - crop_sz)
            y1 = max(0, my - crop_sz)
            x2 = min(TARGET_W, mx + crop_sz)
            y2 = min(new_h, my + crop_sz)
            
            roi = app.clean_frame[y1:y2, x1:x2]
            
            if roi.size > 0:
                # Phóng to ROI (Interpolation Nearest để giữ độ nét pixel)
                zoomed = cv2.resize(roi, (0,0), fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_NEAREST)
                
                # Vẽ tâm chữ thập lên kính lúp
                zh, zw = zoomed.shape[:2]
                cv2.line(zoomed, (zw//2, 0), (zw//2, zh), (0, 0, 255), 1)
                cv2.line(zoomed, (0, zh//2), (zw, zh//2), (0, 0, 255), 1)
                
                # Viền cho kính lúp
                cv2.rectangle(zoomed, (0,0), (zw-1, zh-1), (255, 255, 255), 2)
                
                # Dán kính lúp vào góc Phải-Trên (Top-Right) màn hình để không che chuột
                # (Cách lề 20px)
                margin = 20
                if zw < TARGET_W and zh < new_h:
                    img_show[margin:margin+zh, TARGET_W-margin-zw:TARGET_W-margin] = zoomed
                    cv2.putText(img_show, "ZOOM 4x", (TARGET_W-margin-zw, margin-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Smart Distance", img_show)
        key = cv2.waitKey(1)
        if key == ord('q'): break
        if key == ord('r'):
            app.clicked_points = []
            app.measure_points = []
            app.matrix_homography = None
            app.clean_frame = app.orig_resized.copy()
            print("\n--- ĐÃ RESET ---")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()