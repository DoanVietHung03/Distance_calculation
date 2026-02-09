import cv2
import numpy as np

class ROIStabilizer:
    def __init__(self):
        # Cấu hình Lucas-Kanade Optical Flow
        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        self.old_gray = None    # Frame xám cũ để tracking
        self.p0 = None          # Điểm đặc trưng GỐC (tại thời điểm init)
        self.p_curr = None      # Điểm đặc trưng HIỆN TẠI (đang di chuyển)
        
        self.initial_roi = None # Tọa độ ROI gốc (không bao giờ bị thay đổi)

    def initialize(self, frame, initial_roi_points):
        """
        Khởi tạo Tracker: Lưu lại frame gốc và tìm điểm đặc trưng trên sàn
        """
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.initial_roi = np.array(initial_roi_points, dtype=np.float32)
        
        # Tạo mask để tìm điểm đặc trưng (ưu tiên tìm trong và quanh vùng ROI)
        mask = np.zeros_like(self.old_gray)
        roi_cnt = np.array(initial_roi_points, dtype=np.int32).reshape((-1, 1, 2))
        
        # Mở rộng vùng tìm kiếm ra ngoài ROI một chút để bám chắc hơn
        cv2.fillPoly(mask, [roi_cnt], 255)
        cv2.polylines(mask, [roi_cnt], True, 255, 30) 
        
        # Tìm các điểm tốt nhất để track (Corner detection)
        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=mask, maxCorners=300, qualityLevel=0.01, minDistance=10)
        
        if self.p0 is None:
            # Fallback nếu sàn quá trơn
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, maxCorners=300, qualityLevel=0.01, minDistance=10)
            print("[STABILIZER] Warn: Không tìm thấy đặc trưng trong ROI, track toàn màn hình.")
            
        self.p_curr = self.p0.copy() # Khởi đầu thì điểm hiện tại = điểm gốc
        print(f"[STABILIZER] Initialized with {len(self.p0)} feature points.")

    def update(self, frame):
        """
        Trả về: (ROI mới, Ma trận Homography Global)
        Logic: Track từ frame trước -> frame này, nhưng tính Homography từ Gốc -> Hiện tại
        """
        if self.old_gray is None or self.p0 is None:
            return self.initial_roi, None

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Tính Optical Flow: Điểm cũ (p_curr) đã chạy đi đâu trong frame mới?
        p_new, status, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p_curr, None, **self.lk_params)
        
        # 2. Lọc các điểm track tốt
        if p_new is not None:
            good_new = p_new[status == 1]
            good_old_curr = self.p_curr[status == 1] # Điểm ở frame trước
            good_orig = self.p0[status == 1]         # Điểm tương ứng ở frame GỐC
            
            # Cập nhật cho vòng lặp sau
            self.old_gray = frame_gray.copy()
            self.p_curr = good_new.reshape(-1, 1, 2)
            self.p0 = good_orig.reshape(-1, 1, 2) # Chỉ giữ lại những điểm gốc còn track được
            
            # 3. Tính Homography trực tiếp từ FRAME GỐC (good_orig) -> FRAME HIỆN TẠI (good_new)
            # Đây là chìa khóa để chống trôi (Drift)
            if len(good_new) >= 4:
                M, _ = cv2.findHomography(good_orig, good_new, cv2.RANSAC, 5.0)
                
                if M is not None:
                    # Transform ROI gốc bằng ma trận mới này
                    pts_reshaped = self.initial_roi.reshape(-1, 1, 2)
                    new_roi = cv2.perspectiveTransform(pts_reshaped, M)
                    return new_roi.reshape(-1, 2), M

        # Nếu mất dấu (track fail), trả về vị trí cũ hoặc khởi tạo lại (ở đây giữ nguyên cho đơn giản)
        return self.initial_roi, None