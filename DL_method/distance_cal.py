import contextlib
import cv2
import torch
import numpy as np
from transformers import pipeline
from PIL import Image
import threading
import queue
import time
from ultralytics import YOLO

# ================= CẤU HÌNH NGƯỜI DÙNG =================
VIDEO_PATH = "..\\test_imgs\\cam_2\\cam_2.mp4" 

# 1. HỆ SỐ SCALE
SCALE_FACTOR = 11.06 

# 2. CẤU HÌNH HIỂN THỊ
DISPLAY_WIDTH = 1200  

# 3. THÔNG SỐ CAMERA
FOCAL_LENGTH = 1600 

# Model AI
DEPTH_MODEL_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
YOLO_MODEL_PATH = "..\\weights\\yolo11n.onnx"

# Performance tuning
DEPTH_INPUT_WIDTH = 640
DEPTH_SKIP = 2
DETECT_SKIP = 5                
# ========================================================

# --- CLASS ĐO ĐỘ TRỄ (THREAD-SAFE) ---
class LatencyProfiler:
    def __init__(self):
        self.records = {}
        self.lock = threading.Lock() # Cần Lock vì nhiều thread ghi cùng lúc

    def update(self, name, elapsed_ms):
        with self.lock:
            if name not in self.records: self.records[name] = []
            self.records[name].append(elapsed_ms)

    def print_report(self):
        print("\n" + "="*45)
        print("MULTI-THREAD LATENCY REPORT (Unit: ms)")
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

class Measure3DVideoTool:
    def __init__(self):
        self.profiler = LatencyProfiler() # <--- INIT PROFILER

        self.device = 0 if torch.cuda.is_available() else -1
        print(f"[Init] Đang load model trên {'GPU' if self.device==0 else 'CPU'}...")
        
        # Load Depth Model
        self.pipe = pipeline(task="depth-estimation", model=DEPTH_MODEL_REPO, device=self.device)
        
        # Load Video
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        if not self.cap.isOpened():
            print(f"[ERR] Không mở được video: {VIDEO_PATH}")
            exit()
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or np.isnan(self.fps): 
            self.fps = 30
        
        self.frame_duration_ms = int(1000 / self.fps)
        print(f"[Info] Video FPS: {self.fps} -> Frame duration: {self.frame_duration_ms}ms")

        self.org_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.org_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.scale_ratio = DISPLAY_WIDTH / self.org_w
        self.new_h = int(self.org_h * self.scale_ratio)
        self.new_w = DISPLAY_WIDTH
        
        print(f"[Info] Video gốc: {self.org_w}x{self.org_h} -> Hiển thị: {self.new_w}x{self.new_h}")
        
        self.cx = self.new_w / 2
        self.cy = self.new_h / 2
        
        if FOCAL_LENGTH is None:
            self.fx = self.new_w * 0.9 
            self.fy = self.new_w * 0.9
        else:
            self.fx = FOCAL_LENGTH * self.scale_ratio
            self.fy = FOCAL_LENGTH * self.scale_ratio
            
        self.depth_map = None
        self.current_frame_display = None 
        self.points = [] 
        self.target = None 
        self.is_paused = False 

        # YOLO Config
        self.YOLO_MODEL_PATH = YOLO_MODEL_PATH
        self.YOLO_INPUT_SIZE = 640
        self.YOLO_CONF_THRES = 0.35
        self.yolo_model = None
        try:
            self.yolo_model = YOLO(self.YOLO_MODEL_PATH, task='detect') 
            print(f"[Info] Loaded YOLO model: {self.YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"[Warn] Lỗi load YOLO: {e}")

        # Threading Setup
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=1) 
        self.detect_queue = queue.Queue(maxsize=1) 
        self.persons = [] 

        threading.Thread(target=self.depth_worker, daemon=True).start()
        threading.Thread(target=self.detect_worker, daemon=True).start()

        self.depth_w = min(self.new_w, DEPTH_INPUT_WIDTH)
        self.depth_h = max(1, int(self.new_h * (self.depth_w / float(self.new_w))))
        self.frame_idx = 0

    def push_to_queue(self, q, data):
        try:
            q.put_nowait(data)
        except queue.Full:
            with contextlib.suppress(Exception):
                q.get_nowait()
                q.put_nowait(data)

    def process_depth_frame(self, frame_bgr):
        # Resize và inference
        frame_small = cv2.resize(frame_bgr, (self.depth_w, self.depth_h))
        img_pil = Image.fromarray(cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
        
        depth_output = self.pipe(img_pil)
        depth_tensor = depth_output["predicted_depth"]
        depth_map_raw = depth_tensor.squeeze().cpu().numpy()

        # Resize output map (phần này nhẹ, nhưng cứ tính chung vào depth latency)
        self.depth_map = cv2.resize(depth_map_raw, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)

    def depth_worker(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.2)
                
                # --- ĐO DEPTH ---
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t0 = time.perf_counter()
                
                self.process_depth_frame(frame)
                
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t1 = time.perf_counter()
                self.profiler.update("Depth_Est", (t1 - t0) * 1000)
                # ----------------
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Depth Err: {e}")

    def detect_worker(self):
        while not self.stop_event.is_set():
            try:
                frame_small = self.detect_queue.get(timeout=0.2)
                
                # --- ĐO YOLO ---
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t0 = time.perf_counter()
                
                self.persons = self.detect_persons(frame_small)
                
                if torch.cuda.is_available(): torch.cuda.synchronize()
                t1 = time.perf_counter()
                self.profiler.update("YOLO_Detect", (t1 - t0) * 1000)
                # ---------------

            except queue.Empty:
                continue

    def detect_persons(self, frame_sq):
        out = []
        if self.yolo_model is None or self.depth_map is None: return out
        
        ih, iw = frame_sq.shape[:2]
        results = self.yolo_model(frame_sq, imgsz=self.YOLO_INPUT_SIZE, verbose=False)[0]

        if len(results.boxes) == 0: return out

        sx = self.new_w / float(iw)
        sy = self.new_h / float(ih)
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy()
        dh, dw = self.depth_map.shape[:2]

        for i, box in enumerate(boxes):
            if confs[i] < self.YOLO_CONF_THRES or int(clss[i]) != 0: continue
            x1, y1, x2, y2 = box
            x = int(x1 * sx); y = int(y1 * sy)
            w = int((x2 - x1) * sx); h = int((y2 - y1) * sy)
            foot_x = int(x + w / 2); foot_y = int(y + h)
            foot_x = max(0, min(foot_x, dw - 1))
            foot_y = max(0, min(foot_y, dh - 1))
            
            patch = self.depth_map[max(0, foot_y-2):min(dh, foot_y+3), max(0, foot_x-2):min(dw, foot_x+3)]
            if patch.size == 0: continue
            raw_val = float(np.median(patch))
            c3d = self.pixel_to_3d(foot_x, foot_y, raw_val)
            out.append({'box': (x, y, w, h), 'foot_px': (foot_x, foot_y), 'c3d': c3d, 'conf': confs[i]})
        return out

    def pixel_to_3d(self, u, v, raw_depth_val):
        if raw_depth_val <= 0: return None
        Z = SCALE_FACTOR / raw_depth_val
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        return np.array([X, Y, Z])

    def draw_and_show(self):  # sourcery skip: extract-method
        # Đo thời gian vẽ và hiển thị (Optional)
        # t0 = time.perf_counter()
        if self.current_frame_display is None: return
        img = self.current_frame_display.copy()

        status = "PAUSED" if self.is_paused else "PLAYING"
        cv2.putText(img, f"{status} | FPS: {int(self.fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        for i, (px, py, _) in enumerate(self.points):
            d_val = self.depth_map[py, px] if self.depth_map is not None else 1
            curr_c3d = self.pixel_to_3d(px, py, d_val)
            self.points[i] = (px, py, curr_c3d) 
            cv2.circle(img, (px, py), 5, (0,0,255), -1)
            cv2.putText(img, f"P{i+1}", (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        if self.target:
            tx, ty, _ = self.target
            if self.depth_map is not None:
                d_val_t = self.depth_map[ty, tx]
                self.target = (tx, ty, self.pixel_to_3d(tx, ty, d_val_t))
            cv2.circle(img, (tx, ty), 6, (0,255,255), -1)
            cv2.putText(img, "Target", (tx+8, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        if self.persons:
            p = self.persons[0]
            x, y, w, h = p['box']
            fx, fy = p['foot_px']
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(img, (fx, fy), 5, (0, 255, 0), -1)
            if self.target and self.target[2] is not None and p['c3d'] is not None:
                dist = np.linalg.norm(p['c3d'] - self.target[2])
                cv2.line(img, (fx, fy), (self.target[0], self.target[1]), (255,255,0), 2)
                mid = ((fx + self.target[0])//2, (fy + self.target[1])//2)
                cv2.putText(img, f"{dist:.2f}m", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        if len(self.points) >= 2:
            p1, p2 = self.points[-2], self.points[-1]
            if p1[2] is not None and p2[2] is not None:
                d = np.linalg.norm(p1[2] - p2[2])
                cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (0,255,255), 2)
                mid = ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2)
                cv2.putText(img, f"{d:.2f}m", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Measure 3D", img)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.depth_map is None: return
            val = self.depth_map[y, x]
            c3d = self.pixel_to_3d(x, y, val)
            self.target = (x, y, c3d)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.target = None
            self.points = []

    def run(self):
        cv2.namedWindow("Measure 3D")
        cv2.setMouseCallback("Measure 3D", self.mouse_callback)
        print("Ready. Press Space to Pause/Play, Q to Quit.")

        while True:
            start_time = time.time()
            if not self.is_paused:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                self.frame_idx += 1
                
                # --- ĐO MAIN LOOP (Display FPS) ---
                # Chỉ đo thời gian resize + hiển thị
                loop_t0 = time.perf_counter()

                display_frame = cv2.resize(frame, (self.new_w, self.new_h))
                self.current_frame_display = display_frame

                if self.frame_idx % DEPTH_SKIP == 0:
                    self.push_to_queue(self.frame_queue, frame)

                if self.frame_idx % DETECT_SKIP == 0:
                    small_frame = cv2.resize(frame, (self.YOLO_INPUT_SIZE, self.YOLO_INPUT_SIZE))
                    self.push_to_queue(self.detect_queue, small_frame)

                self.draw_and_show()
                
                loop_t1 = time.perf_counter()
                self.profiler.update("Main_Display", (loop_t1 - loop_t0) * 1000)
                # ----------------------------------

            if self.is_paused:
                wait_ms = 30
            else:
                process_time_ms = int((time.time() - start_time) * 1000)
                wait_ms = max(1, self.frame_duration_ms - process_time_ms)

            key = cv2.waitKey(wait_ms)
            if key == ord('q'): 
                self.profiler.print_report() # <--- IN BÁO CÁO KHI THOÁT
                break
            if key == ord(' '): self.is_paused = not self.is_paused

        self.stop_event.set()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    Measure3DVideoTool().run()