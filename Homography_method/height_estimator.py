import numpy as np
import json
import math
import cv2

class HeightEstimator:
    def __init__(self):
        self.fx = 0.0
        self.fy = 0.0
        self.loaded = False

    def load_focal_length(self, calib_file, current_width):
        """
        Đọc Focal Length từ file JSON và scale theo kích thước ảnh hiện tại.
        """
        try:
            with open(calib_file, 'r') as f:
                data = json.load(f)
            
            K = np.array(data['camera_matrix'])
            
            # Lấy độ phân giải gốc để tính tỉ lệ scale
            if 'image_resolution' in data:
                orig_w, orig_h = data['image_resolution']
            else:
                orig_w = 1280 # Fallback nếu không có thông tin
            
            # Tính tỉ lệ resize
            scale_ratio = current_width / orig_w
            
            # Scale tiêu cự (fx, fy)
            self.fx = K[0, 0] * scale_ratio
            self.fy = K[1, 1] * scale_ratio
            self.loaded = True
            
            print(f"[HeightEstimator] Focal Length loaded: fx={self.fx:.1f}, fy={self.fy:.1f}")
            return True
        except Exception as e:
            print(f"[HeightEstimator] ERR: {e}")
            # Fallback giá trị ước lượng (thường bằng chiều rộng ảnh)
            self.fx = current_width
            self.fy = current_width
            return False

    def calculate(self, head_pt, foot_pt, homography_matrix):
        """
        Tính chiều cao thực tế dựa trên công thức Pinhole + Sliding Scale.
        Output: (Chiều cao mét, Khoảng cách tới camera mét)
        """
        if not self.loaded or homography_matrix is None:
            return 0.0, 0.0

        # 1. Tính chiều cao trên ảnh (Pixel)
        # Dùng khoảng cách Euclide giữa đầu và chân trên ảnh
        h_pixel = math.hypot(head_pt[0] - foot_pt[0], head_pt[1] - foot_pt[1])

        # 2. Tính khoảng cách thực từ Camera tới điểm Chân (Distance D)
        # Convert điểm chân pixel sang tọa độ thực sàn nhà
        foot_arr = np.array([[[foot_pt[0], foot_pt[1]]]], dtype=np.float32)
        real_pt = cv2.perspectiveTransform(foot_arr, homography_matrix)[0][0] # (X_real, Y_real)
        
        # Giả định Camera nằm tại gốc (0,0) hoặc tính khoảng cách từ điểm click đầu tiên
        # Distance = sqrt(x^2 + y^2)
        distance_D = math.sqrt(real_pt[0]**2 + real_pt[1]**2)

        # 3. Áp dụng công thức Sliding Scale: H_real = h_pixel * (D / f)
        # Dùng fy (tiêu cự dọc)
        if self.fy == 0: return 0.0, 0.0
        
        height_real = h_pixel * (distance_D / self.fy)

        # Hệ số điều chỉnh góc nghiêng (Tilt Correction) - Tùy chỉnh
        TILT_FACTOR = 1.0 
        
        return height_real * TILT_FACTOR, distance_D