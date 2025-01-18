import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('model.h5')

classes = {
    1: 'Tốc độ tối đa cho phép là (20km/h)',
    2: 'Tốc độ tối đa cho phép là (30km/h)',
    3: 'Tốc độ tối đa cho phép là (50km/h)',
    4: 'Tốc độ tối đa cho phép là (60km/h)',
    5: 'Tốc độ tối đa cho phép là (70km/h)',
    6: 'Tốc độ tối đa cho phép là (80km/h)',
    7: 'Hết hạn chế tốc độ tối đa là (80km/h)',
    8: 'Tốc độ tối đa cho phép là (100km/h)',
    9: 'Tốc độ tối đa cho phép là (120km/h)',
    10: 'Cấm vượt',
    11: 'Cấm vượt xe > 3.5 tấn',
    12: 'Giao nhau với đường không ưu tiên',
    13: 'Bắt đầu đoạn đường ưu tiên',
    14: 'Giao nhau với đường ưu tiên',
    15: 'Dừng lại',
    16: 'Đường cấm',
    17: 'Cấm xe > 3.5 tấn',
    18: 'Cấm đi ngược chiều',
    19: 'Nguy hiểm khác',
    20: 'Chỗ ngoặt nguy hiểm vòng bên trái',
    21: 'Chỗ ngoặt nguy hiểm vòng bên phải',
    22: 'Nhiều chỗ ngoặc nguy hiểm liên tiếp',
    23: 'Đường lòi lõm',
    24: 'Đường trơn',
    25: 'Đường bị hẹp về bên phải',
    26: 'Công trường thi công',
    27: 'Giao nhau có tín hiệu đèn',
    28: 'Đường dành cho người đi bộ cắt ngang',
    29: 'Trẻ em',
    30: 'Đường người đi xe đạp cắt ngang',
    31: 'Cẩn thận băng giá/tuyết',
    32: 'Thú rừng vượt qua đường',
    33: 'Hết tất cả các lệnh cấm',
    34: 'Các xe chỉ được rẻ phải',
    35: 'Các xe chỉ được rẻ trái',
    36: 'Các xe chỉ được đi thẳng',
    37: 'Các xe chỉ thẳng hoặc rẽ phải',
    38: 'Các xe chỉ thẳng hoặc rẽ trái',
    39: 'Hướng đi vòng chướng ngại vật sang phải',
    40: 'Hướng đi vòng chướng ngại vật sang trái',
    41: 'Nơi giao nhau chạy theo vòng xuyến',
    42: 'Hết cấm vượt',
    43: 'Hết cấm vượt xe > 3.5 tấn'
}

top = tk.Tk()

window_width = 600
window_height = 650

screen_width = top.winfo_screenwidth()
screen_height = top.winfo_screenheight()

center_x = int(screen_width / 2 - window_width / 2)
center_y = int(screen_height / 2 - window_height / 2 - 40)

top.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
top.title('Nhận dạng biển báo giao thông')
top.configure(background='#f0f0f0')

title_font = ('Helvetica', 20, 'bold')
subtitle_font = ('Helvetica', 14)
button_font = ('Helvetica', 12, 'bold')
label_font = ('Helvetica', 16, 'bold')

heading = Label(top, text="Nhận dạng biển báo giao thông", pady=10, font=title_font, background='#f0f0f0', foreground='#333333')
heading1 = Label(top, text="Môn Học: Artificial intelligence", pady=10, font=subtitle_font, background='#f0f0f0', foreground='#333333')
heading2 = Label(top, text="Nguyễn Nhất Huy - 21200294", pady=5, font=subtitle_font, background='#f0f0f0', foreground='#333333')
heading3 = Label(top, text="Nguyễn Giang - 21200284", pady=5, font=subtitle_font, background='#f0f0f0', foreground='#333333')

label = Label(top, background='#f0f0f0', font=label_font)
sign_image = Label(top)

def preprocess_image(img):
    img = cv2.resize(img, (30, 30))  # Thay đổi kích thước ảnh về 30x30
    img = img / 255.0  # Chuẩn hóa ảnh về dải [0, 1]
    return img

def detect_regions(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask1 + mask2

    lower_blue = np.array([90, 80, 0])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = mask_red + mask_blue
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def classify_region(roi, model, threshold):
    roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    img_preprocessed = preprocess_image(np.array(roi_pil))
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

    predictions = model.predict(img_preprocessed)
    classIndex = np.argmax(predictions, axis=1)[0]
    probabilityValue = np.max(predictions)

    if probabilityValue > threshold:
        return classes.get(classIndex + 1, "Không nhận dạng được"), probabilityValue
    else:
        return None, None

def process_frame(frame, model, threshold):
    detected_regions = []
    filtered_contours = []

    for cnt in detect_regions(frame):
        area = cv2.contourArea(cnt)
        if area > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            overlap = False
            for x2, y2, w2, h2 in detected_regions:
                if (x < x2 + w2 and x + w > x2 and y < y2 + h2 and y + h > y2):
                    overlap = True
                    break

            if not overlap:
                filtered_contours.append(cnt)
                detected_regions.append((x, y, w, h))

    detected_signs = []
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = frame[y:y + h, x:x + w]
        label, probability = classify_region(roi, model, threshold)
        if probability:
            detected_signs.append(label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 205, 50), 2)

    return frame, detected_signs

def classify(file_path):
    global label_packed
    try:
        image = cv2.imread(file_path)
        image_with_signs, detected_signs = process_frame(image, model, threshold=0.95)
        image_with_signs = cv2.cvtColor(image_with_signs, cv2.COLOR_BGR2RGB)
        image_with_signs = Image.fromarray(image_with_signs)

        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 3), (top.winfo_height() / 3)))
        imgtk = ImageTk.PhotoImage(image_with_signs.resize(uploaded.size))

        sign_image.configure(image=imgtk)
        sign_image.image = imgtk

        if detected_signs:
            text_var.set('\n'.join(detected_signs))
            result_label.config(foreground='#32cd32')
        else:
            text_var.set('Không nhận dạng được biển báo giao thông')
            result_label.config(foreground='red')

        label.configure(foreground='#011638')
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể phân loại biển báo: {str(e)}")

def show_classify_button(file_path):
    classify_b = Button(top, text="Nhận dạng", command=lambda: classify(file_path), padx=10, pady=5, background='#4CAF50', foreground='white', font=button_font)
    classify_b.place(relx=0.6, rely=0.85)
    upload.place(relx=0.2, rely=0.85)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        if file_path:
            uploaded = Image.open(file_path)
            uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
            imgtk = ImageTk.PhotoImage(uploaded)

            sign_image.configure(image=imgtk)
            sign_image.image = imgtk
            label.configure(text='')

            show_classify_button(file_path)
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể tải ảnh: {str(e)}")

upload = Button(top, text="Tải ảnh lên", command=upload_image, padx=10, pady=5, background='#007BFF', foreground='white', font=button_font)
upload.pack(side=tk.BOTTOM, pady=50)

heading.pack()
heading1.pack()
heading2.pack()
heading3.pack()
sign_image.pack()
label.pack()

text_var = StringVar()
result_label = Label(top, textvariable=text_var, pady=20, font=label_font, background='#f0f0f0', foreground='#333333')
result_label.pack()

top.mainloop()
