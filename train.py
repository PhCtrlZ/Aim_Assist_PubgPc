from ultralytics import YOLO
import os

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load model
    model = YOLO('yolov8s.pt')  # hoặc yolov8l.pt nếu muốn chính xác nhất

    # Train
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,          # giảm xuống 8 hoặc 4 nếu GPU yếu
        device=0,          # 0 = GPU, 'cpu' nếu không có GPU
        patience=20,
        name='pubg_detect',
        cache=True,
        workers=4          # số worker cho dataloader
    )

    print("✅ Training completed!")
    print(f"Model saved at: runs/detect/pubg_detect/weights/best.pt")

if __name__ == '__main__':
    main()