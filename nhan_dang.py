import cv2
import numpy as np

def empty(a):
    pass

# Đọc ảnh gốc
img = cv2.imread("duong_thang_01.png")

if img is None:
    print("Không đọc được ảnh! Hãy kiểm tra lại đường dẫn.")
    exit()

# Tiền xử lý
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Tạo cửa sổ thanh trượt
cv2.namedWindow("Thanh dieu chinh")
cv2.resizeWindow("Thanh dieu chinh", 640, 450)

# THANH TRƯỢT ĐƯỜNG THẲNG
cv2.createTrackbar("Nguong Canny 1", "Thanh dieu chinh", 50, 255, empty)
cv2.createTrackbar("Nguong Canny 2", "Thanh dieu chinh", 150, 255, empty)
cv2.createTrackbar("Nguong duong thang", "Thanh dieu chinh", 50, 200, empty)
cv2.createTrackbar("Do dai duong thang", "Thanh dieu chinh", 50, 300, empty)
cv2.createTrackbar("Khoang cach duong", "Thanh dieu chinh", 10, 100, empty)

# THANH TRƯỢT ĐƯỜNG TRÒN
cv2.createTrackbar("Khoang cach hinh tron", "Thanh dieu chinh", 30, 200, empty)
cv2.createTrackbar("Nguong Canny tron", "Thanh dieu chinh", 100, 255, empty)
cv2.createTrackbar("Nguong phat hien tron", "Thanh dieu chinh", 30, 150, empty)
cv2.createTrackbar("Ban kinh nho","Thanh dieu chinh", 10, 150, empty)
cv2.createTrackbar("Ban kinh lon", "Thanh dieu chinh", 200, 400, empty)

print("Keo thanh truot de dieu chinh.")
print("Nhan phim 'q' de luu va thoat.")

while True:
    img_copy = img.copy()
    try:
        # LẤY THAM SỐ ĐƯỜNG THẲNG
        c_t1 = cv2.getTrackbarPos("Nguong Canny 1", "Thanh dieu chinh")
        c_t2 = cv2.getTrackbarPos("Nguong Canny 2", "Thanh dieu chinh")
        l_thresh = max(1, cv2.getTrackbarPos("Nguong duong thang","Thanh dieu chinh"))
        min_len = cv2.getTrackbarPos("Do dai duong thang", "Thanh dieu chinh")
        max_gap = cv2.getTrackbarPos("Khoang cach duong", "Thanh dieu chinh")

        # LẤY THAM SỐ ĐƯỜNG TRÒN
        md = max(1,cv2.getTrackbarPos("Khoang cach hinh tron","Thanh dieu chinh"))
        p1 = max(1,cv2.getTrackbarPos("Nguong Canny tron", "Thanh dieu chinh"))
        p2 = max(1,cv2.getTrackbarPos("Nguong phat hien tron", "Thanh dieu chinh"))
        min_r = cv2.getTrackbarPos("Ban kinh nho", "Thanh dieu chinh")
        max_r = cv2.getTrackbarPos("Ban kinh lon", "Thanh dieu chinh")
    except cv2.error:
        break

    # 1. NHẬN DẠNG ĐƯỜNG THẲNG
    edges = cv2.Canny(blur, c_t1, c_t2)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=l_thresh, minLineLength=min_len, maxLineGap=max_gap)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.flatten()
            cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 2. NHẬN DẠNG ĐƯỜNG TRÒN
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=md, param1=p1, param2=p2, minRadius=min_r, maxRadius=max_r)

    if circles is not None:
        for x, y, r in np.round(
            circles[0, :]
        ).astype(int):
            # Viền hình tròn màu đỏ
            cv2.circle( img_copy, (x, y), r, (0, 0, 255), 2)
            # Tâm hình tròn màu xanh
            cv2.circle(img_copy, (x, y), 3, (255, 0, 0),-1)

    # HIỂN THỊ
    scale = 0.35
    img_show = cv2.resize( img_copy, None, fx=scale, fy=scale)
    cv2.imshow( "Ket qua realtime", img_show)

    # Nhấn q để lưu
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("results_google/test06_result.jpg",img_copy)
        print("Da luu ket qua: ""results_google/test06_result.jpg")
        break

cv2.destroyAllWindows()