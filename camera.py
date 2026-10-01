import cv2
import mediapipe as mp
import numpy as np
import math
import os

# 1. MEDIA PIPE - NHẬN DIỆN BÀN TAY
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 2. MỞ CAMERA
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Khong mo duoc camera!")
    exit()

# 3. TẠO THƯ MỤC LƯU KẾT QUẢ

# 4. BIẾN
points = []
result_text = ""
so_anh = 1

# 5. NHẬN DẠNG ĐƯỜNG THẲNG
def la_duong_thang(points):
    if len(points) < 20:
        return False
    pts = np.array(points, dtype=np.float32)
    [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx = float(vx)
    vy = float(vy)
    x0 = float(x0)
    y0 = float(y0)

    distances = []

    for x, y in pts:
        distance = abs(vy * (x - x0) - vx * (y - y0))
        distances.append(distance)
    mean_distance = np.mean(distances)
    return mean_distance < 20

# 6. NHẬN DẠNG HÌNH TRÒN
def la_hinh_tron(points):
    if len(points) < 30:
        return False
    pts = np.array(points, dtype=np.float32)
    (cx, cy), radius = cv2.minEnclosingCircle(pts)
    distances = []
    for x, y in pts:
        d = math.sqrt( (x - cx) ** 2 + (y - cy) ** 2)
        distances.append(d)
    distances = np.array(distances)
    mean_radius = np.mean(distances)
    if mean_radius == 0:
        return False
    std_radius = np.std(distances)
    error = std_radius / mean_radius
    # Điểm đầu và điểm cuối
    start = pts[0]
    end = pts[-1]
    distance_start_end = math.sqrt( (start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
    # Kiểm tra đường có khép kín không
    closed = distance_start_end < mean_radius * 0.7
    # Kiểm tra độ tròn
    round_shape = error < 0.30
    return closed and round_shape

# 7. NHẬN DẠNG HÌNH
def nhan_dang(points):
    if len(points) < 20:
        return "CHUA DU DIEM"

    # Kiểm tra hình tròn trước
    if la_hinh_tron(points):
        return "HINH TRON"

    # Kiểm tra đường thẳng
    if la_duong_thang(points):
        return "DUONG THANG"
    return "KHONG XAC DINH"

# 8. HIỂN THỊ HÌNH CHUẨN
def hien_thi_hinh_chuan(frame, result):
    h, w, _ = frame.shape
    # Vị trí hình chuẩn
    center_x = int(w * 0.75)
    center_y = int(h * 0.70)

    # HÌNH TRÒN
    if result == "HINH TRON":
        cv2.circle(frame, (center_x, center_y), 100, (0, 0, 255), 5)
        cv2.putText(frame, "HINH TRON", (center_x - 100, center_y + 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ĐƯỜNG THẲNG
    elif result == "DUONG THANG":
        cv2.line(frame,(center_x - 120, center_y), (center_x + 120, center_y), (0, 0, 255), 5)
        cv2.putText(frame,"DUONG THANG",(center_x - 120, center_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# 9. VÒNG LẶP CAMERA
while True:
    ret, frame = cap.read()
    if not ret:
        print("Khong doc duoc camera!")
        break
    # Lật camera giống gương
    frame = cv2.flip(frame, 1)
    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe xử lý
    result = hands.process(rgb)

    # Biến xác định có đang vẽ hay không
    drawing_mode = False

    # 10. CHỈ THEO DÕI NGÓN TRỎ
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Ngón trỏ
            index_tip = hand_landmarks.landmark[8]
            index_pip = hand_landmarks.landmark[6]

            # Ngón giữa
            middle_tip = hand_landmarks.landmark[12]
            middle_pip = hand_landmarks.landmark[10]

            # Ngón áp út
            ring_tip = hand_landmarks.landmark[16]
            ring_pip = hand_landmarks.landmark[14]

            # Ngón út
            pinky_tip = hand_landmarks.landmark[20]
            pinky_pip = hand_landmarks.landmark[18]
            h, w, _ = frame.shape

            # Tọa độ đầu ngón trỏ
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            # 11. KIỂM TRA TƯ THẾ NGÓN TAY
            # Ngón trỏ duỗi
            index_up = index_tip.y < index_pip.y

            # Ba ngón còn lại gập
            middle_folded = middle_tip.y > middle_pip.y

            ring_folded = ring_tip.y > ring_pip.y

            pinky_folded = pinky_tip.y > pinky_pip.y

            # Điều kiện để bắt đầu vẽ
            drawing_mode = (index_up and middle_folded and ring_folded and pinky_folded)

            # 12. CHỈ HIỂN THỊ ĐẦU NGÓN TRỎ
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

            # 13. LƯU ĐƯỜNG ĐI CỦA NGÓN TRỎ
            if drawing_mode:
                points.append((x, y))

    # 14. HIỂN THỊ TRẠNG THÁI
    if drawing_mode:
        cv2.putText(frame, "DANG VE", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText( frame, "GAP 3 NGON DE DUNG", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 15. VẼ ĐƯỜNG ĐI CỦA NGÓN TRỎ
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 3)

    # 16. NHẬN PHÍM
    key = cv2.waitKey(1) & 0xFF

    # 17. NHẤN SPACE → NHẬN DẠNG
    if key == 32:
        result_text = nhan_dang(points)

    # 18. HIỂN THỊ KẾT QUẢ
    if result_text != "":
        cv2.putText(frame, "Ket qua: " + result_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0),3)
        # Hiển thị hình chuẩn
        hien_thi_hinh_chuan(frame, result_text)

    # 19. HƯỚNG DẪN
    cv2.putText(frame, "SPACE: Nhan dang | Q: Xoa | ESC: Thoat", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # 20. LƯU ẢNH KẾT QUẢ
    if key == 32:
        if result_text == "HINH TRON":
            ten_file = (f"hinh_tron_{so_anh:02d}.png")
        elif result_text == "DUONG THANG":
            ten_file = (f"duong_thang_{so_anh:02d}.png")
        else:
            ten_file = (f"khong_xac_dinh_{so_anh:02d}.png")
        # Lưu ảnh
        cv2.imwrite(ten_file, frame)
        print("Da luu anh:", ten_file)
        so_anh += 1

    # 21. HIỂN THỊ CAMERA
    cv2.imshow("Ve hinh bang ngon tro", frame)

    # 22. NHẤN Q → XÓA
    if key == ord('q'):
        points = []
        result_text = ""

    # 23. ESC → THOÁT
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
