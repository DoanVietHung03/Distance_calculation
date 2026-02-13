import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ================= NHẬP SỐ LIỆU CỦA BẠN VÀO ĐÂY =================
# Ví dụ: Đây là số liệu giả định (bạn hãy thay bằng số đo thật)
data = {
    'Real_Distance': [1.44, 2.79, 2.65, 2.6],
    
    # Kết quả đo từ phương pháp Hybrid (Homography)
    'Hybrid_Dist':   [1.58, 2.49, 2.39, 3.2], 
    
    # Kết quả đo từ phương pháp Deep Learning (Depth Anything)
    'DL_Dist':       [0.88, 2.9, 3.01, 2.5]
}
# ================================================================

df = pd.DataFrame(data)

# 1. Tính toán sai số
def calculate_metrics(real, measured):
    ae = np.abs(measured - real)
    re = (ae / real) * 100
    return ae, re

df['Hybrid_AE'], df['Hybrid_RE'] = calculate_metrics(df['Real_Distance'], df['Hybrid_Dist'])
df['DL_AE'], df['DL_RE'] = calculate_metrics(df['Real_Distance'], df['DL_Dist'])

# 2. In bảng báo cáo
print("=== BẢNG TỔNG HỢP SAI SỐ (DISTANCE) ===")
print(df[['Real_Distance', 'Hybrid_Dist', 'Hybrid_RE', 'DL_Dist', 'DL_RE']].round(2))

print("\n=== KẾT LUẬN ===")
print(f"Hybrid MAPE: {df['Hybrid_RE'].mean():.2f}%")
print(f"DL MAPE:     {df['DL_RE'].mean():.2f}%")

# 3. Vẽ biểu đồ so sánh (Để đưa vào báo cáo)
plt.figure(figsize=(10, 6))

# Vẽ đường sai số phần trăm
plt.plot(df['Real_Distance'], df['Hybrid_RE'], marker='o', label='Hybrid (Homography)', linewidth=2)
plt.plot(df['Real_Distance'], df['DL_RE'], marker='s', label='Deep Learning', linewidth=2)

plt.title('So sánh Sai số theo Khoảng cách (Distance Error Rate)', fontsize=14)
plt.xlabel('Khoảng cách thực tế (m)', fontsize=12)
plt.ylabel('Sai số tương đối (%)', fontsize=12)
plt.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='Ngưỡng chấp nhận (5%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Lưu ảnh
plt.savefig('error_chart.png')
print("\n[INFO] Đã lưu biểu đồ vào file 'error_chart.png'")