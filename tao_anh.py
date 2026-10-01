import math
import cv2
import numpy as np

# Tạo ảnh nền đen kích thước 600x800
img = np.zeros((600, 800, 3), dtype=np.uint8)

# Vẽ 2 đường thẳng
cv2.line(img, (100, 100), (700, 100), (255, 255, 255), 3)
cv2.line(img, (150, 500), (650, 200), (255, 255, 255), 3)

# Vẽ 2 hình tròn
cv2.circle(img, (250, 350), 80, (255, 255, 255), 3)
cv2.circle(img, (550, 400), 50, (255, 255, 255), 3)

# Vẽ 2 hình chữ nhật
cv2.rectangle(img, (50, 50), (200, 150), (255, 255, 255), 3)
cv2.rectangle(img, (500, 300), (700, 500), (255, 255, 255), 3)

# Vẽ ảnh đường lượn sóng
for x in range(50, 750):
    y = int(550 + 30 * math.sin(x * 0.05))
    
    if x > 50:
        cv2.line(img, (x - 1, old_y), (x, y), (255, 255, 255), 3)
    
    old_y = y

# Đường nét đứt
y = 250

for x in range(50, 750, 80):
    cv2.line( img, (x, y), (x + 40, y), (255, 255, 255), 3)

#Thêm một số nhiễu vào ảnh
noise_img = img.copy()

# Số lượng điểm nhiễu
amount = 5000
for i in range(amount):
    # Tọa độ ngẫu nhiên
    x = np.random.randint(0, 800)
    y = np.random.randint(0, 600)
    # 50% điểm trắng, 50% điểm đen
    if np.random.rand() < 0.5:
        noise_img[y, x] = (255, 255, 255)
    else:
        noise_img[y, x] = (0, 0, 0)

# Tạo ảnh xám và làm mờ để nhận dạng đường thẳng và đường tròn
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Vẽ ảnh có độ nhiễu
cv2.imshow("Anh co do nhieu", blur)

# Lưu ảnh
cv2.imwrite("test06.png", img)

cv2.waitKey(0)
cv2.destroyAllWindows()