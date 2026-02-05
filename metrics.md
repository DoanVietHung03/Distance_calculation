**Thông số camera iphone 12 thường**
- Hệ thống camera kép độ phân giải 12MP: Camera Chính và Ultra Wide
- Camera chính: khẩu độ (độ mở của ống kính → quyết định lượng ánh sáng đi vào cảm biến) ƒ/1.6 và FOV ~75-80°
- Camera Ultra Wide: khẩu độ ƒ/2.4 và FOV 120°
- Len 1 và 0.5

**Yêu cầu cơ bản**
- Các vật thể cần nhận diện khoảng cách cần phải nằm chung trong 1 mặt phẳng 2D, không được đưa vào yếu tố về chiều cao, chiều sâu.
- Cần biết chi tiết kích thước thực tế (chiều dài, chiều rộng và độ dài đường chéo của đa giác), càng chính xác thì kết quả trả về có sai số càng bé.
- Ta cần chấm điểm ROI trên ảnh chính xác nhất có thể với điểm mà ta đã chọn ở ngoài thực tế và đã đo khoảng cách thực tế đó.
- Do đây là bài toán quy định về 1 mặt phẳng 2D nên yếu tố len của camera không còn quan trọng nữa, có thể bỏ qua trong tính toán.

**Test cases**
1) Khoảng cách ~6m trong văn phòng (chạy 10 lần):
- chiều rộng từ điểm cửa đến góc nhọn: 1.83m
- chiều dài từ góc nhọn đến hộp tai nghe: 3.72m
- chiều rộng từ hộp tai nghe đến chân tủ: 1.28m
- chiều dài từ chân tủ đến cửa: 4.74m
- đường chéo cửa đến hộp tai nghe: 4.9m
- Khoảng cách cần tính: ~2.5m
- len cam: 1
    -> Kết quả: trung bình ~2.438m (sai số trung bình dao động trong khoảng +-10cm)
- len cam: 0.5m
    -> Kết quả: trung bình ~2.4m (sai số trung bình dao động trong khoảng +-12cm)

2) Khoảng cách ~15m ngoài trời (chạy 10 lần):
- chiều rộng từ hộp tai nghe đến đuôi xe đen: 3.45m
- chiều dài từ đuôi xe đen đến bọc nylon: 10.74m
- chiều rộng từ bọc nylon đến nắp ly nước: 3.2m
- chiều dài từ nắp ly nước đến hộp tai nghe: 12m
- đường chéo hộp tai nghe đến bọ nylon: 11.54m
- Khoảng cách cần tính: ~3.26m
- len cam: 1
    -> Kết quả: trung bình ~3.44m (sai số trung bình dao động trong khoảng +-30cm)
- len cam: 0.5m
    -> Kết quả: trung bình ~3.38m (sai số trung bình dao động trong khoảng +-40cm)