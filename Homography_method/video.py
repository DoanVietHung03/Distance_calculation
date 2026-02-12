import contextlib
import cv2
import numpy as np
import math
import os
import json
import time
from ultralytics import YOLO
import torch
import threading
import queue

# IMPORT MODULE CŨ
from height_estimator import HeightEstimator

# ================= CẤU HÌNH =================
VIDEO_PATH = '..\\test_imgs\\cam_2\\cam_2.mp4'
CALIB_FILE = '..\\calibration.json'
CONFIG_FILE = 'config.json'
TARGET_W = 1200
YOLO_SKIP_FRAMES = 2 # Giảm skip xuống vì giờ đã có thread riêng, detect tích cực hơn cũng được

# ================= PROFILER (THREAD-SAFE) =================
class LatencyProfiler:
    def __init__(self):
        self.records = {}
        self.lock = threading.Lock()

    def update(self, name, elapsed_ms):
        with self.lock:
            if name not in self.records: self.records[name] = []
            self.records[name].append(elapsed_ms)

    def print_report(self):
        print("\n" + "="*45)
        print("MULTI-THREAD VIDEO STABILIZER REPORT (Unit: ms)")
        print("="*45)
        print(f"{'Component':<15} | {'Mean':<8} | {'Min':<8} | {'Max':<8} | {'P99':<8}")
        print("-" * 60)
        
        for name, values in self.records.items():
            if not values: continue
            arr = np.array(values)
            if len(arr) > 5: arr = arr[5:] # Bỏ warm-up
            
            mean_v = np.mean(arr)
            min_v = np.min(arr)
            max_v = np.max(arr)
            p99_v = np.percentile(arr, 99)
            
            print(f"{name:<15} | {mean_v:.2f}     | {min_v:.2f}     | {max_v:.2f}     | {p99_v:.2f}")
        print("="*45 + "\n")

# ================= VIDEO STREAM (GIỮ NGUYÊN) =================
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or math.isnan(self.fps): self.fps = 30
        self.delay = 1.0 / self.fps

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            start_time = time.time()
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.ret = False
                time.sleep(0.01)
                continue
            self.ret = True
            self.frame = frame
            elapsed = time.time() - start_time
            time_to_wait = self.delay - elapsed
            if time_to_wait > 0:
                time.sleep(time_to_wait)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True   

# ================= YOLO WORKER (CLASS MỚI) =================
class YoloWorker:
    def __init__(self, model_path, profiler):
        self.profiler = profiler
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.stopped = False
        self.model_path = model_path
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        
        # Khởi tạo model trong thread riêng hoặc main đều được, 
        # nhưng tốt nhất là init trước khi start thread
        self.model = YOLO(self.model_path)
        print(f"[INFO] YOLO Worker loaded on {self.device}")
        
        # Start Thread
        threading.Thread(target=self.run, daemon=True).start()

    def put_frame(self, frame):
        if not self.input_queue.full():
            self.input_queue.put_nowait(frame)

    def get_results(self):
        return None if self.output_queue.empty() else self.output_queue.get_nowait()

    def run(self):
        while not self.stopped:
            try:
                frame = self.input_queue.get(timeout=0.1)

                # --- ĐO YOLO LATENCY ---
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t0 = time.perf_counter()

                results = self.model(frame, verbose=False, device=self.device, conf=0.5, imgsz=640)

                if torch.cuda.is_available(): torch.cuda.synchronize()
                t1 = time.perf_counter()
                self.profiler.update("YOLO_Infer", (t1 - t0) * 1000)
                if self.output_queue.full():
                    # Nếu main thread chưa kịp lấy kết quả cũ, vứt đi, lấy cái mới nhất
                    with contextlib.suppress(queue.Empty):
                        self.output_queue.get_nowait()
                self.output_queue.put(results)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[YOLO Thread Error] {e}")

    def stop(self):
        self.stopped = True

# ================= MAIN APP =================
class VideoDistanceApp:
    def __init__(self):
        self.profiler = LatencyProfiler()
        self.mode = "DISTANCE"
        self.paused = False
        self.frame_count = 0

        # --- STABILIZER CONFIG ---
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.gray_anchor = None
        self.p0_anchor = None
        self.roi_points_initial = None
        self.roi_points_curr = None
        self.target_point_initial = None
        self.target_point_curr = None
        self.last_known_boxes = [] 
        self.target_tracker = None

        # Tools
        self.height_tool = HeightEstimator()
        
        # Khởi tạo YOLO Worker
        self.yolo_worker = YoloWorker('..\\weights\\yolo11n-pose.onnx', self.profiler)

        # Data
        self.real_world = {}
        self.cam_real_pos = (0.5, -18.0)
        self.clicked_points_orig = []
        self.matrix_homography = None
        self.scale_px_per_meter = 1.0
        self.map1, self.map2 = None, None

        # Runtime
        self.current_frame = None
        self.detected_objects = [] # List này sẽ được update bất đồng bộ
        self.prev_time = 0
        self.fps = 0

    def init_calibration_maps(self, original_size):
        if not os.path.exists(CALIB_FILE): return
        try:
            with open(CALIB_FILE, 'r') as f: data = json.load(f)
            K = np.array(data['camera_matrix'])
            D = np.array(data['distortion_coefficients'])
            w_orig, h_orig = original_size
            scale_factor = TARGET_W / w_orig
            target_h = int(h_orig * scale_factor)
            target_size = (TARGET_W, target_h)

            if 'image_resolution' in data:
                calib_w, calib_h = data['image_resolution']
                total_scale_x = (w_orig / calib_w) * scale_factor
                total_scale_y = (h_orig / calib_h) * scale_factor
                K[0, 0] *= total_scale_x; K[1, 1] *= total_scale_y
                K[0, 2] *= total_scale_x; K[1, 2] *= total_scale_y
            else:
                K[0, 0] *= scale_factor; K[1, 1] *= scale_factor
                K[0, 2] *= scale_factor; K[1, 2] *= scale_factor

            new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, target_size, 1, target_size)
            self.map1, self.map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, target_size, 5)
            self.height_tool.load_focal_length(CALIB_FILE, TARGET_W)
        except Exception as e:
            print(f"Error Init Calib: {e}")

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
                    self.roi_points_initial = np.array(self.clicked_points_orig, dtype=np.float32).reshape(-1, 1, 2)
                    self.roi_points_curr = self.roi_points_initial.copy()
                    self.compute_homography(self.roi_points_curr)
                if 'target_point' in data:
                    tp = data['target_point']
                    self.target_point_initial = np.array([[tp]], dtype=np.float32)
                    self.target_point_curr = self.target_point_initial.copy()
                return True
        except Exception:
            return False

    def get_quadrilateral_coords(self, l1, l2, l3, l4, d13):
        # (Giữ nguyên logic toán học cũ)
        if l1 + l2 < d13 or abs(l1 - l2) > d13: return []
        p1 = (0.0, 0.0); p2 = (l1, 0.0)
        cos_alpha = (l1**2 + d13**2 - l2**2) / (2 * l1 * d13)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        alpha = math.acos(cos_alpha)
        p3 = (d13 * math.cos(alpha), d13 * math.sin(alpha))
        d = d13; a = (l4**2 - l3**2 + d**2) / (2*d)
        h = math.sqrt(max(0, l4**2 - a**2))
        rx = -(p3[1] - p1[1]) / d; ry = (p3[0] - p1[0]) / d
        x0 = p1[0] + a * (p3[0] - p1[0]) / d; y0 = p1[1] + a * (p3[1] - p1[1]) / d
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
        with contextlib.suppress(Exception):
            self.matrix_homography = cv2.getPerspectiveTransform(src_pts, dst_pts)

    def calculate_distance_points(self, p1, p2):
        if self.matrix_homography is None: return 0.0
        pts = np.float32([p1, p2]).reshape(-1, 1, 2)
        trans_pts = cv2.perspectiveTransform(pts, self.matrix_homography)
        dist_px = np.linalg.norm(trans_pts[0][0] - trans_pts[1][0])
        return dist_px / self.scale_px_per_meter

    # ================= ANCHOR STABILIZER =================
    def init_anchor(self, gray_frame):
        self.gray_anchor = gray_frame.copy()
        mask = np.ones_like(gray_frame, dtype=np.uint8) * 255
        if self.roi_points_initial is not None:
            pts = self.roi_points_initial.astype(np.int32)
            cv2.fillPoly(mask, [pts], 0) 
        if len(self.last_known_boxes) > 0:
            for box in self.last_known_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        self.p0_anchor = cv2.goodFeaturesToTrack(self.gray_anchor, mask=mask, maxCorners=300, qualityLevel=0.01, minDistance=10)

    def update_stabilizer(self, gray_curr):
        if self.p0_anchor is None or len(self.p0_anchor) < 10:
            self.init_anchor(gray_curr)
            return

        p1, st, err = cv2.calcOpticalFlowPyrLK(self.gray_anchor, gray_curr, self.p0_anchor, None, **self.lk_params)
        
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.p0_anchor[st == 1]
            
            # Filter occlusion (chỉ giữ lại điểm không nằm đè lên người)
            if len(self.last_known_boxes) > 0:
                boxes_np = np.array(self.last_known_boxes)
                x1 = boxes_np[:, 0] - 10; y1 = boxes_np[:, 1] - 10
                x2 = boxes_np[:, 2] + 10; y2 = boxes_np[:, 3] + 10
                pts_x = good_new[:, 0]; pts_y = good_new[:, 1]
                in_x_range = (pts_x[:, None] >= x1) & (pts_x[:, None] <= x2)
                in_y_range = (pts_y[:, None] >= y1) & (pts_y[:, None] <= y2)
                is_dirty_point = np.any(in_x_range & in_y_range, axis=1)
                clean_new = good_new[~is_dirty_point]
                clean_old = good_old[~is_dirty_point]
            else:
                clean_new = good_new
                clean_old = good_old

            if len(clean_new) > 10:
                M, mask_ransac = cv2.findHomography(clean_old, clean_new, cv2.RANSAC, 5.0)
                if M is not None:
                    if self.roi_points_initial is not None:
                        self.roi_points_curr = cv2.perspectiveTransform(self.roi_points_initial, M)
                        self.compute_homography(self.roi_points_curr)
                    if self.target_point_initial is not None:
                        self.target_point_curr = cv2.perspectiveTransform(self.target_point_initial, M)

            track_ratio = len(clean_new) / (len(self.p0_anchor) + 1e-5)
            if track_ratio < 0.3:
                self.init_anchor(gray_curr)

    def get_current_target_tuple(self):
        if self.target_point_curr is not None:
            return tuple(self.target_point_curr[0][0].astype(int))
        return None

    def update_target_tracker(self, frame):
        track_box = None
        if self.target_tracker is not None:
            success, box = self.target_tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                pt = (x + w//2, y + h//2)
                self.target_point_curr = np.array([[pt]], dtype=np.float32)
                track_box = (x, y, w, h)
            else:
                self.target_tracker = None
        return track_box

    # ================= LOGIC XỬ LÝ KẾT QUẢ TỪ THREAD =================
    def process_async_detections(self):
        """Kiểm tra xem Thread YOLO có trả kết quả mới về không"""
        results = self.yolo_worker.get_results()
        if results is None: return # Không có kết quả mới, giữ nguyên detected_objects cũ

        # Nếu có kết quả mới, parse và update logic
        # Lưu ý: Kết quả này là của frame N, bây giờ video có thể đang ở frame N+2.
        # Tuy nhiên do góc quay camera thay đổi không quá nhanh, ta chấp nhận độ trễ này.
        
        self.detected_objects = []
        self.last_known_boxes = [] 
        target_pt = self.get_current_target_tuple()

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            kpts_data = None
            if r.keypoints is not None and r.keypoints.data is not None:
                kpts_data = r.keypoints.data.cpu().numpy()

            for i, box in enumerate(boxes):
                self.last_known_boxes.append(box) # Update để Stabilizer né ở frame sau
                x1, y1, x2, y2 = box
                head_point = (int((x1 + x2) / 2), y1)
                ground_point = (int((x1 + x2) / 2), y2)

                if kpts_data is not None and len(kpts_data) > i:
                    kp = kpts_data[i]
                    if kp[15][2] > 0.5 and kp[16][2] > 0.5:
                        gx = (kp[15][0] + kp[16][0]) / 2
                        gy = (kp[15][1] + kp[16][1]) / 2
                        ground_point = (int(gx), int(gy))
                    elif kp[15][2] > 0.5:
                        ground_point = (int(kp[15][0]), int(kp[15][1]))
                    elif kp[16][2] > 0.5:
                        ground_point = (int(kp[16][0]), int(kp[16][1]))
                    
                    if kp[0][2] > 0.5:
                        head_point = (int(kp[0][0]), int(kp[0][1]))
                    elif kp[1][2] > 0.5 and kp[2][2] > 0.5:
                        hx = (kp[1][0] + kp[2][0]) / 2
                        hy = (kp[1][1] + kp[2][1]) / 2
                        head_point = (int(hx), int(hy))

                # Tính toán logic ngay tại đây (trên Main Thread, nhưng data từ Async)
                # Dùng matrix_homography hiện tại (Mới nhất) cho Box (Cũ) -> Chấp nhận được
                obj_info = {
                    'box': box, 'head': head_point, 'foot': ground_point, 
                    'h_real': 0.0, 'd_to_target': 0.0
                }
                
                if self.mode == "HEIGHT":
                    h_real, _ = self.height_tool.calculate(head_point, ground_point, self.matrix_homography, self.cam_real_pos)
                    obj_info['h_real'] = h_real
                elif self.mode == "DISTANCE" and target_pt:
                    d_target = self.calculate_distance_points(ground_point, target_pt)
                    obj_info['d_to_target'] = d_target
                
                self.detected_objects.append(obj_info)

    def process_frame(self, raw_frame):
        # 1. Đo Loop Hiển thị (Display Latency)
        t_start = time.perf_counter()
        
        curr_time = time.time()
        self.fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = curr_time
        
        h, w = raw_frame.shape[:2]
        scale = TARGET_W / w
        new_h = int(h * scale)
        frame_resized = cv2.resize(raw_frame, (TARGET_W, new_h))
        
        if self.map1 is not None:
            frame_clean = cv2.remap(frame_resized, self.map1, self.map2, cv2.INTER_LINEAR)
        else:
            frame_clean = frame_resized
            
        frame_gray = cv2.cvtColor(frame_clean, cv2.COLOR_BGR2GRAY)

        # 2. Update Stabilizer (Optical Flow - Chạy trên Main Thread, BẮT BUỘC)
        self.profiler.update("Stabilizer", 0) # Placeholder start
        t_stab = time.perf_counter()
        
        if self.gray_anchor is None:
            self.init_anchor(frame_gray)
        else:
            self.update_stabilizer(frame_gray)
            
        self.profiler.update("Stabilizer", (time.perf_counter() - t_stab) * 1000)
            
        target_box = self.update_target_tracker(frame_clean)
        
        # 3. Gửi Frame cho Worker Thread (Không chặn main thread)
        if self.frame_count % YOLO_SKIP_FRAMES == 0:
            self.yolo_worker.put_frame(frame_clean.copy()) # Copy để tránh race condition memory

        # 4. Kiểm tra kết quả trả về từ Worker (Nếu có thì update self.detected_objects)
        self.process_async_detections()
        
        self.profiler.update("Main_Display", (time.perf_counter() - t_start) * 1000)
        
        self.frame_count += 1
        return frame_clean, target_box

    def draw_overlays(self, img, target_box):
        cv2.putText(img, f"FPS: {int(self.fps)}", (TARGET_W - 120, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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

        status_text = "PAUSED" if self.paused else "PLAYING"
        cv2.rectangle(img, (0, 0), (TARGET_W, 60), (0, 0, 0), -1)
        cv2.putText(img, f"MODE: {self.mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(img, status_text, (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if self.paused else (0, 255, 0), 1)

app = VideoDistanceApp()

def mouse_event_video(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and app.mode == "DISTANCE":
        app.target_point_curr = np.array([[[x, y]]], dtype=np.float32)
        box_size = 20
        bbox = (max(0, x - box_size//2), max(0, y - box_size//2), box_size, box_size)
        with contextlib.suppress(Exception):
            app.target_tracker = cv2.TrackerKCF_create()
            app.target_tracker.init(app.current_frame, bbox)

def main():
    vs = VideoStream(VIDEO_PATH).start()
    time.sleep(1.0)

    width = int(vs.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vs.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    app.init_calibration_maps((width, height))
    app.load_config()

    cv2.namedWindow("Anchor Tracking")
    cv2.setMouseCallback("Anchor Tracking", mouse_event_video)

    while True:
        if not app.paused:
            ret, frame = vs.read()
            if not ret:
                print("End of video, resetting...")
                vs.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                app.gray_anchor = None 
                time.sleep(0.5)
                continue
            app.current_frame, target_box = app.process_frame(frame)
        else:
            target_box = None
            if app.current_frame is None: 
                ret, frame = vs.read()
                if ret:
                    app.current_frame, target_box = app.process_frame(frame)

        if app.current_frame is not None:
            display_img = app.current_frame.copy()
            app.draw_overlays(display_img, target_box)
            cv2.imshow("Anchor Tracking", display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            app.yolo_worker.stop() # Dừng thread
            app.profiler.print_report()
            vs.stop()
            break
        if key == ord(' '): app.paused = not app.paused
        if app.paused:
            if key == ord('h'): app.mode = "HEIGHT"
            if key == ord('d'): app.mode = "DISTANCE"

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()