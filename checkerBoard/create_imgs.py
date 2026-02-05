import cv2
import os

def extract_frames_from_video(video_path, output_folder, frame_interval=30):
    """
    Cắt ảnh từ video.
    :param video_path: Đường dẫn đến file video (mp4, avi...)
    :param output_folder: Thư mục để lưu ảnh.
    :param frame_interval: Cứ sau bao nhiêu frame thì lưu 1 tấm (để tránh ảnh trùng lặp).
                           Ví dụ: Video 30fps, để interval=15 nghĩa là 0.5 giây lấy 1 tấm.
    """
    
    # 1. Tạo thư mục output nếu chưa có
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[INFO] Đã tạo thư mục: {output_folder}")

    # 2. Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Không thể mở file video: {video_path}")
        return

    frame_count = 0
    saved_count = 0
    
    print("[INFO] Đang xử lý... Nhấn Ctrl+C để dừng sớm nếu muốn.")

    while True:
        ret, frame = cap.read()
        
        # Hết video thì dừng
        if not ret:
            break

        # Chỉ lưu frame nếu chia hết cho interval
        if frame_count % frame_interval == 0:
            filename = f"calib_{saved_count:03d}.jpg"
            save_path = os.path.join(output_folder, filename)
            
            cv2.imwrite(save_path, frame)
            print(f" -> Đã lưu: {filename}")
            saved_count += 1
        
        frame_count += 1

    cap.release()
    print("------------------------------------------------")
    print(f"[XONG] Tổng cộng đã lưu {saved_count} tấm ảnh vào thư mục '{output_folder}'.")

# ==========================================
# CẤU HÌNH CHẠY
# ==========================================
if __name__ == "__main__":
    # 1. Tên file video bạn đã quay (nhớ để cùng thư mục hoặc điền đúng đường dẫn)
    VIDEO_FILE = "checkerBoard.mp4" 
    
    # 2. Thư mục muốn lưu ảnh
    OUTPUT_DIR = "calibration_images"
    
    # 3. Tần suất lấy ảnh (Video 30fps thì để 15-30 là đẹp)
    INTERVAL = 20 

    extract_frames_from_video(VIDEO_FILE, OUTPUT_DIR, INTERVAL)