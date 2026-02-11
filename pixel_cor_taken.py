import cv2

VIDEO_PATH = ".\\test_imgs\\cam_2\\cam_2.mp4"
TARGET_WIDTH = 1200
WINDOW_NAME = "Click to get pixel coordinates"

points = []
frame = None  # đảm bảo là biến global

def resize_keep_ratio(img, target_width):
    h, w = img.shape[:2]
    scale = target_width / w
    new_h = int(h * scale)
    resized = cv2.resize(img, (target_width, new_h))
    return resized, scale


def mouse_callback(event, x, y, flags, param):
    global frame, points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        text = f"{len(points)}: ({x}, {y})"

        cv2.putText(
            frame, text, (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 1
        )

        print(text)
        cv2.imshow(WINDOW_NAME, frame)


def main():
    global frame

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Không đọc được video")
        return

    frame, scale = resize_keep_ratio(frame, TARGET_WIDTH)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # ⚠️ RẤT QUAN TRỌNG
    cv2.imshow(WINDOW_NAME, frame)     # render trước
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("👉 Click chuột trái để lấy tọa độ")
    print("👉 Nhấn q hoặc ESC để thoát")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

    print("\n📌 Các điểm đã chọn:")
    for i, p in enumerate(points, 1):
        print(f"{i}: {p}")

    print("\nMap về ảnh gốc:")
    print("x0 = x / scale, y0 = y / scale")


if __name__ == "__main__":
    main()