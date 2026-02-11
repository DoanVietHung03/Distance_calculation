import cv2
import torch
import numpy as np
from transformers import pipeline
from PIL import Image
import threading
import queue
import time

# ================= CẤU HÌNH NGƯỜI DÙNG =================
VIDEO_PATH = "..\\test_imgs\\cam_2\\cam_2.mp4"  # <--- ĐỔI ĐƯỜNG DẪN FILE VIDEO CỦA BẠN

# 1. HỆ SỐ SCALE (Lấy từ bước Calibrate)
SCALE_FACTOR = 12.72 

# 2. CẤU HÌNH HIỂN THỊ (RESIZE)
DISPLAY_WIDTH = 1200  # Giảm xuống nếu video giật lag

# 3. THÔNG SỐ CAMERA (PINHOLE)
# Nếu không biết, để None code tự ước lượng
FOCAL_LENGTH = 1600 

# Model AI
DEPTH_MODEL_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
# ========================================================

class Measure3DVideoTool:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"[Init] Đang load model trên {'GPU' if self.device==0 else 'CPU'}...")
        
        # Load Model
        self.pipe = pipeline(task="depth-estimation", model=DEPTH_MODEL_REPO, device=self.device)
        
        # Load Video
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        if not self.cap.isOpened():
            print(f"[ERR] Không mở được video: {VIDEO_PATH}")
            exit()

        # Lấy thông số video gốc
        self.org_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.org_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # --- XỬ LÝ RESIZE ---
        # Tính tỉ lệ resize để hiển thị
        self.scale_ratio = DISPLAY_WIDTH / self.org_w
        self.new_h = int(self.org_h * self.scale_ratio)
        self.new_w = DISPLAY_WIDTH
        
        print(f"[Info] Video gốc: {self.org_w}x{self.org_h} -> Hiển thị: {self.new_w}x{self.new_h}")
        
        # Cập nhật thông số Pinhole theo kích thước hiển thị (new_w, new_h)
        self.cx = self.new_w / 2
        self.cy = self.new_h / 2
        
        if FOCAL_LENGTH is None:
            self.fx = self.new_w * 0.9 
            self.fy = self.new_w * 0.9
        else:
            self.fx = FOCAL_LENGTH * self.scale_ratio
            self.fy = FOCAL_LENGTH * self.scale_ratio
            
        self.depth_map = None
        self.current_frame_display = None # Lưu frame hiện tại để vẽ UI
        self.points = [] # Danh sách điểm click
        self.is_paused = False # Trạng thái Pause
        # Threading/queue for background depth computation
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.depth_thread = threading.Thread(target=self.depth_worker, daemon=True)
        self.depth_thread.start()

    def process_depth_frame(self, frame_bgr):
        """Tính Depth Map cho 1 frame"""
        # Resize frame trước khi đưa vào model để đồng bộ với hiển thị và tăng tốc độ
        frame_resized = cv2.resize(frame_bgr, (self.new_w, self.new_h))
        
        # Chuyển sang PIL
        img_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        
        # Inference
        depth_output = self.pipe(img_pil)
        depth_tensor = depth_output["predicted_depth"]
        depth_map_raw = depth_tensor.squeeze().cpu().numpy()
        
        # Resize lại depth map cho chắc chắn khớp 100% kích thước hiển thị
        self.depth_map = cv2.resize(depth_map_raw, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)
        self.current_frame_display = frame_resized

    def depth_worker(self):
        """Background worker that consumes frames and computes depth."""
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self.process_depth_frame(frame)
            except Exception as e:
                print(f"[Worker ERR] {e}")
            finally:
                try:
                    self.frame_queue.task_done()
                except Exception:
                    pass

    def pixel_to_3d(self, u, v, raw_depth_val):
        """Chuyển đổi Pixel (u,v) + Depth -> Tọa độ 3D (X, Y, Z)"""
        if raw_depth_val <= 0: return None
        
        # 1. Tính Z (Mét)
        Z = SCALE_FACTOR / raw_depth_val
        
        # 2. Tính X, Y (Mét) dùng công thức Pinhole
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        
        return np.array([X, Y, Z])

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.depth_map is None: return

            # Lấy giá trị depth tại thời điểm click
            raw_val = self.depth_map[y, x]
            coord_3d = self.pixel_to_3d(x, y, raw_val)
            
            if coord_3d is not None:
                self.points.append((x, y, coord_3d))
                print(f"Click [{len(self.points)}]: Pixel({x},{y}) -> DepthVal: {raw_val:.2f} -> Z: {coord_3d[2]:.2f}m")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Chuột phải để Reset
            self.points = []
            print("[Reset] Đã xóa các điểm đo.")

    def draw_and_show(self):
        if self.current_frame_display is None: return

        img_display = self.current_frame_display.copy()
        
        # Hiển thị trạng thái PAUSE
        if self.is_paused:
            cv2.putText(img_display, "PAUSED (Click to Measure)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(img_display, "PLAYING (Press Space to Pause)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Vẽ các điểm đã click
        # Lưu ý: Với Video, các điểm này ghim theo tọa độ màn hình (pixel). 
        # Nếu camera di chuyển, vị trí thực tế sẽ thay đổi, nhưng code này đang đo realtime theo pixel đó.
        for i, (px, py, old_c3d) in enumerate(self.points):
            # Cập nhật lại Z theo frame hiện tại (nếu muốn realtime distance khi video chạy)
            # Hoặc giữ nguyên giá trị lúc click (ở đây ta tính lại theo frame hiện tại để thấy khoảng cách thay đổi)
            curr_depth_val = self.depth_map[py, px]
            curr_c3d = self.pixel_to_3d(px, py, curr_depth_val)
            
            # Cập nhật lại giá trị 3D trong list points (để tính khoảng cách mới)
            self.points[i] = (px, py, curr_c3d)

            color = (0, 0, 255) if i == 0 else (255, 0, 0)
            cv2.circle(img_display, (px, py), 5, color, -1)
            cv2.putText(img_display, f"P{i+1}", (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Tính khoảng cách giữa 2 điểm cuối
        if len(self.points) >= 2:
            p1 = self.points[-2]
            p2 = self.points[-1]
            
            xyz1 = p1[2]
            xyz2 = p2[2]
            
            dist_3d = np.linalg.norm(xyz1 - xyz2)
            
            cv2.line(img_display, (p1[0], p1[1]), (p2[0], p2[1]), (0, 255, 255), 2)
            
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            
            label = f"{dist_3d:.2f}m"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img_display, (mid_x, mid_y - h - 5), (mid_x + w, mid_y + 5), (0, 0, 0), -1)
            cv2.putText(img_display, label, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Measure 3D Video", img_display)

    def run(self):
        cv2.namedWindow("Measure 3D Video")
        cv2.setMouseCallback("Measure 3D Video", self.mouse_callback)
        
        print("\n--- HƯỚNG DẪN ---")
        print("1. SPACE: Tạm dừng / Tiếp tục video")
        print("2. Chuột TRÁI: Chọn điểm đo")
        print("3. Chuột PHẢI: Xóa điểm")
        print("4. 'q': Thoát")

        while True:
            # Nếu KHÔNG PAUSE thì đọc frame mới và xử lý
            if not self.is_paused:
                ret, frame = self.cap.read()
                if not ret:
                    print("Hết video hoặc lỗi đọc frame. Loop lại từ đầu...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                # Enqueue frame for background depth computation.
                # If the queue is full, replace the previous frame (drop-old behavior).
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        # remove old frame and put the new one
                        _ = self.frame_queue.get_nowait()
                        self.frame_queue.task_done()
                    except Exception:
                        pass
                    try:
                        self.frame_queue.put_nowait(frame)
                    except Exception:
                        pass
            
            # Vẽ giao diện (dù pause hay play đều vẽ lại để hiển thị điểm click)
            self.draw_and_show()
            
            key = cv2.waitKey(1 if not self.is_paused else 30)
            
            if key == ord('q'):
                break
            elif key == ord(' '): # Phím Space
                self.is_paused = not self.is_paused
                state = "PAUSED" if self.is_paused else "PLAYING"
                print(f"--- {state} ---")

        # signal worker to stop and wait
        self.stop_event.set()
        self.depth_thread.join(timeout=1.0)
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tool = Measure3DVideoTool()
    tool.run()