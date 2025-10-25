from ultralytics import YOLO
import cv2

# Load model
model = YOLO('runs/detect/pubg_detect4/weights/best.pt')

# Xem các class
print("Classes:", model.names)
print()

# Test trên ảnh
img_path = 'test.jpg'  # đổi tên file nếu cần
results = model.predict(img_path, conf=0.35, save=True, show=True)

# In chi tiết detection
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        xyxy = box.xyxy[0].tolist()
        print(f"Class: {model.names[cls]}, Confidence: {conf:.2f}, Box: {xyxy}")