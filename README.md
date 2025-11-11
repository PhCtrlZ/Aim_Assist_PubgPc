# Aim_Assist_PubgPc
# YOLOv8 — Human Shape Detection
> This repository shows how to train YOLOv8 to detect humans (person / human silhouette).  
> **Purpose:** research / academic / benign CV applications (surveillance, people counting, robotics).  
> **Not** for cheating, targeting, or other malicious uses.

---

## 🚀 Overview

Mục tiêu: huấn luyện một mô hình YOLOv8 nhận diện **người (person / human shapes)** từ ảnh/video. README này hướng dẫn từ chuẩn bị dữ liệu, annotation, cấu hình dataset tới lệnh huấn luyện, đánh giá và chạy inference.

---

## ⚙️ Yêu cầu

- Python 3.8+
- GPU được khuyến nghị (CUDA + cuDNN) để huấn luyện
- Bộ cài cần thiết:
```bash
pip install -U pip
pip install ultralytics opencv-python tqdm matplotlib seaborn
# nếu muốn annotate local: pip install labelme
.
├── data/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── configs/
│   └── dataset.yaml
├── notebooks/           # (tùy chọn) notebook cho EDA, inference tests
├── scripts/
│   ├── visualize.py
│   └── inference.py
├── README.md
└── requirements.txt
class_id center_x center_y width height
0 0.5123 0.4321 0.1234 0.3456
path: ../data   # root path to images/ and labels/
train: images/train
val: images/val
test: images/test  # optional

names:
  0: person
# từ thư mục chứa dataset.yaml
# ví dụ chọn model yolov8n (nano)
yolo task=detect mode=train model=yolov8n.pt data=configs/dataset.yaml epochs=50 imgsz=640 batch=16
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(data='configs/dataset.yaml', epochs=50, imgsz=640, batch=16)
# validate
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=configs/dataset.yaml
# scripts/inference.py
import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

img = 'data/images/test/0001.jpg'
results = model.predict(source=img, imgsz=640, conf=0.25, iou=0.45)

# Hiển thị kết quả
res = results[0]
img_out = res.plot()  # trả về numpy image with boxes
cv2.imshow('result', img_out[:,:,::-1])  # BGR<->RGB
cv2.waitKey(0)
cv2.destroyAllWindows()
python scripts/inference.py
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=data/images/test/0001.jpg
