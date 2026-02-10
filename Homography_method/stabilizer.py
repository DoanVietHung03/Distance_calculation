import cv2
import numpy as np

class PerspectiveStabilizer:
    def __init__(self):
        # Cấu hình Lucas-Kanade Optical Flow
        # winSize: Kích thước cửa sổ tìm kiếm
        # maxLevel: Số tầng tháp ảnh (pyramid levels)
        self.lk_params = dict(winSize=(21, 21),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))
        
        self.p0 = None          # Điểm đặc trưng ở Frame Gốc
        self.gray0 = None       # Ảnh xám Frame Gốc
        self.current_matrix = np.eye(3, dtype=np.float32) # Ma trận biến đổi hiện tại

    def initialize(self, frame, roi_points):
        """
        Input: roi_points (4 điểm sàn) -> Dùng để TẠO MASK loại bỏ vùng sàn
        Ta chỉ track cột nhà, mái che (vùng tĩnh).
        """
        self.gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Tạo Mask: Chọn vùng NỀN (Background), loại bỏ vùng SÀN (ROI)
        # Vì người sẽ đi vào vùng sàn, nếu track sàn sẽ bị sai.
        mask = np.ones_like(self.gray0) * 255
        
        # Tô đen vùng ROI (sàn) để không tìm điểm track ở đó
        roi_cnt = np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [roi_cnt], 0) 
        
        # 2. Tìm điểm đặc trưng tốt ở vùng Background (cột, xe đỗ, mái...)
        self.p0 = cv2.goodFeaturesToTrack(self.gray0, mask=mask, maxCorners=200, qualityLevel=0.01, minDistance=10)
        
        if self.p0 is not None:
            print(f"[STABILIZER] Initialized. Tracking {len(self.p0)} background points.")
        else:
            print("[STABILIZER] ERR: Không tìm thấy điểm nền để track!")

    def update(self, frame):
        """
        Trả về Ma trận biến đổi M (3x3) từ Frame Gốc -> Frame Hiện tại
        """
        if self.p0 is None:
            return np.eye(3, dtype=np.float32) # Trả về ma trận đơn vị nếu lỗi

        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Track điểm từ Frame Gốc sang Frame Hiện tại
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.gray0, gray_curr, self.p0, None, **self.lk_params)
        
        # Lọc các điểm track tốt
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            
            # Nếu còn đủ điểm để tính toán
            if len(good_new) >= 4:
                # Tính Ma trận Homography: Biến đổi từ Gốc (old) -> Hiện tại (new)
                M, _ = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
                
                if M is not None:
                    self.current_matrix = M
                    return M

        # Nếu track thất bại, trả về ma trận cũ hoặc đơn vị
        return self.current_matrix