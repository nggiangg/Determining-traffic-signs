##############
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

#############################################

frameWidth = 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75         # PROBABILITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# IMPORT THE TRAINED MODEL
model = load_model('model.h5')

def preprocessing(img):
    img = img.resize((30, 30))  # Resize image to the input shape of the model
    img = np.asarray(img) / 255.0  # Convert image to numpy array and normalize
    return img

def getClassName(classNo):
    classNames = {
    0: 'Toc do toi da cho phep la (20km/h)',
    1: 'Toc do toi da cho phep la (30km/h)',
    2: 'Toc do toi da cho phep la (50km/h)',
    3: 'Toc do toi da cho phep la (60km/h)',
    4: 'Toc do toi da cho phep la (70km/h)',
    5: 'Toc do toi da cho phep la (80km/h)',
    6: 'Het han che toc do toi da la (80km/h)',
    7: 'Toc do toi da cho phep la (100km/h)',
    8: 'Toc do toi da cho phep la (120km/h)',
    9: 'Cam vuot',
    10: 'Cam vuot xe > 3,5 tan',
    11: 'Giao nhau voi duong khong uu tien',
    12: 'Bat dau doan duong uu tien',
    13: 'Giao nhau voi duong uu tien',
    14: 'Dung lai',
    15: 'Duong cam',
    16: 'Cam xe > 3.5 tan',
    17: 'Cam di nguoc chieu',
    18: 'Nguy hiem khac',
    19: 'Cho ngoac nguy hiem vong ben trai',
    20: 'Cho ngoac nguy hiem vong ben phai',
    21: 'Nhieu cho ngoac nguy hiem lien tiep',
    22: 'Duong loi lom',
    23: 'Duong tron',
    24: 'Duong bi hep ve ben phai',
    25: 'Cong truong thi cong',
    26: 'Giao nhau co tin hieu den',
    27: 'Duong danh cho nguoi di bo cat ngang',
    28: 'Tre em',
    29: 'Duong nguoi di xe dap cat ngang',
    30: 'Can than bang gia/tuyet',
    31: 'Thu rung vuot qua duong',
    32: 'Het tat ca cac lenh cam',
    33: 'Cac xe chi duoc re phai',
    34: 'Cac xe chi duoc re trai',
    35: 'Cac xe chi duoc di thang',
    36: 'Cac xe chi thang hoac re phai',
    37: 'Cac xe chi thang hoac re trai',
    38: 'Huong di vong chuong ngai vat sang phai',
    39: 'Huong di vong chuong ngai vat sang trai',
    40: 'Noi giao nhau chay theo vong xuyen',
    41: 'Het cam vuot',
    42: 'Het cam vuot xe > 3.5 tan'
    }
    return classNames.get(classNo, 'Unknown')
# ... (các import và khai báo khác)

def preprocess_image(img):
    # Chuyển đổi từ PIL Image sang NumPy array
    img = np.array(img, dtype=np.uint8)
    # Chuyển đổi từ RGB sang BGR (OpenCV sử dụng BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (30, 30))
    img = img / 255.0
    return img


def detect_regions(img):
    """Phát hiện các vùng tiềm năng chứa biển báo dựa trên màu sắc."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Định nghĩa dải màu cho đỏ và xanh (đã điều chỉnh)
    lower_red = np.array([0, 40, 40])
    upper_red = np.array([15, 255, 255])
    lower_red2 = np.array([165, 40, 40])
    upper_red2 = np.array([180, 255, 255])

    lower_blue = np.array([90, 80, 0])
    upper_blue = np.array([140, 255, 255])

    # Tạo mặt nạ và tìm contours
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask1 + mask2
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = mask_red + mask_blue

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def classify_region(roi, model, threshold):
    """Phân loại vùng ảnh đã cho bằng mô hình và trả về thông tin nếu vượt quá ngưỡng."""
    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    img_preprocessed = preprocess_image(roi_pil)
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
    # Dự đoán và trích xuất thông tin
    predictions = model.predict(img_preprocessed)
    classIndex = np.argmax(predictions, axis=1)[0]
    # Lấy giá trị xác suất cao nhất
    probabilityValue = np.max(predictions)
    # Trả về thông tin nếu xác suất lớn hơn ngưỡng
    if probabilityValue > threshold:
        return getClassName(classIndex), probabilityValue
    else:
        return None, None

def process_frame(frame, model, threshold, min_area=500, max_ratio_deviation=0.2):  # Thêm tham số max_ratio_deviation
    """Processes a frame, detects square-like signs, and draws non-overlapping results."""

    detected_regions = []

    for cnt in detect_regions(frame):
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)

            # Kiểm tra tỉ lệ giữa chiều rộng và chiều cao
            aspect_ratio = float(w) / h
            if abs(1 - aspect_ratio) <= max_ratio_deviation:  # Chỉ xử lý khi tỉ lệ gần bằng 1

                # Kiểm tra trùng lặp (giữ nguyên)
                overlap = False
                for x2, y2, w2, h2 in detected_regions:
                    if (x < x2 + w2 and x + w > x2 and
                            y < y2 + h2 and y + h > y2):
                        overlap = True
                        break

                if not overlap:
                    roi = frame[y:y + h, x:x + w]
                    label, probability = classify_region(roi, model, threshold)

                    if probability:
                        text = f"{label} ({probability * 100:.2f}%)"
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, text, (x, y - 10), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                        detected_regions.append((x, y, w, h))

    return frame


# ... (các import và các hàm preprocess_image, detect_regions, classify_region, process_frame đã được định nghĩa)



# Cấu hình camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Giảm độ phân giải
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Các biến khác
threshold = 0.95
font = cv2.FONT_HERSHEY_SIMPLEX

# Vòng lặp chính
while True:
    success, imgOriginal = cap.read()
    if not success:
        print("Failed to capture image")
        break

    imgOriginal = process_frame(imgOriginal, model, threshold)    # Xử lý frame và nhận diện biển báo
    cv2.imshow("Result", imgOriginal)  # Hiển thị kết quả

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Thoát khi nhấn 'q'
        break

cap.release()
cv2.destroyAllWindows()