import cv2
import numpy as np

class ROIStabilizer:
    def __init__(self):
        # Cấu hình Lucas-Kanade Optical Flow
        # winSize: Kích thước cửa sổ tìm kiếm
        # maxLevel: Số tầng tháp ảnh (pyramid levels)
        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        self.old_gray = None    # Frame xám cũ
        self.p0 = None          # Điểm đặc trưng cần track
        self.anchor_point = None # Tọa độ điểm neo (để vẽ debug)

    def initialize(self, frame, anchor_point):
        """
        Khởi tạo Tracker tại điểm neo (Anchor Point).
        anchor_point: Tuple (x, y) - Thường là điểm click đầu tiên của ROI.
        """
        self.old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.anchor_point = anchor_point

        # 1. Tạo mask đen toàn bộ
        mask = np.zeros_like(self.old_gray)
        
        # 2. Chỉ khoét lỗ trắng quanh điểm neo (Bán kính 60px)
        # Mục đích: Chỉ tìm đặc trưng ở vật tĩnh (cột, góc tường), tránh bắt vào người đi bộ.
        cv2.circle(mask, anchor_point, 60, 255, -1)
        
        # 3. Tìm các điểm đặc trưng tốt (Góc cạnh mạnh)
        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=mask, maxCorners=50, qualityLevel=0.01, minDistance=5)
        
        # Fallback: Nếu điểm neo quá trơn (sàn gạch bóng), mở rộng vùng tìm kiếm
        if self.p0 is None:
            print("[STABILIZER] Warn: Điểm neo trơn, mở rộng vùng tìm kiếm...")
            cv2.circle(mask, anchor_point, 120, 255, -1)
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=mask, maxCorners=50, qualityLevel=0.01, minDistance=5)
            
        if self.p0 is not None:
            print(f"[STABILIZER] Locked on Anchor {anchor_point} with {len(self.p0)} points.")
        else:
            print("[STABILIZER] ERR: Không tìm thấy điểm track nào! Chống rung sẽ không hoạt động.")

    def update(self, frame):
        """
        Tính toán độ dịch chuyển giữa frame hiện tại và frame trước đó.
        Trả về: (dx, dy)
        """
        if self.old_gray is None or self.p0 is None:
            return 0.0, 0.0

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Tính Optical Flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
        
        dx, dy = 0.0, 0.0
        
        # Chỉ lấy những điểm track thành công (status == 1)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            
            # Cập nhật frame cũ và điểm cũ cho vòng lặp sau
            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
            
            if len(good_new) > 0:
                # Tính độ dịch chuyển trung bình: New - Old
                # Nếu ảnh dịch sang phải (New > Old) -> dx dương
                shift = np.mean(good_new - good_old, axis=0)
                dx, dy = shift[0], shift[1]

        return dx, dy