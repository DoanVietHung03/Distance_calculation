import cv2
import os

# ====== CẤU HÌNH ======
VIDEO_PATH = ".\\test_imgs\\cam_2\\cam_2.mp4"     # đường dẫn tới file .mp4
OUTPUT_DIR = ".\\test_imgs\\cam_2"        # thư mục lưu ảnh
FRAME_INDEX = 1            # số thứ tự frame cần trích (0-based)
OUTPUT_NAME = "frame_test.jpg"

# =====================

def extract_frame(video_path, frame_index, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError(f"Không mở được video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎞 Tổng số frame: {total_frames}")

    if frame_index >= total_frames:
        raise ValueError("FRAME_INDEX vượt quá số frame của video")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()

    if not ret:
        raise RuntimeError("Không đọc được frame")

    cv2.imwrite(output_path, frame)
    cap.release()

    print(f"✅ Đã lưu frame {frame_index} tại: {output_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)

    extract_frame(VIDEO_PATH, FRAME_INDEX, output_path)