from ultralytics import YOLO
import os
import time
from pathlib import Path

MODEL = "yolo12n.pt"
DATA = "data.yaml"

if __name__ == "__main__":
    print("=== Bắt đầu huấn luyện YOLOv12 ===")
    model = YOLO(MODEL)
    results = model.train(
        data=DATA,
        imgsz=640,
        epochs=100,
        batch=8,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        device=0,
        workers=3,
        project="runs_yolo",
        name="train_waid_cfg",
        exist_ok=True,
        cache=True
    )
    # Kiểm tra file model đã tồn tại
    out_dir = Path(results.save_dir)
    best_model = out_dir / "weights" / "best.pt"
    while not best_model.exists():
        print("Chờ ghi file best.pt...")
        time.sleep(5)

    print("=== Huấn luyện và ghi log hoàn tất. Sẽ tắt máy sau 60 giây... ===")
    time.sleep(15)  
    os.system("shutdown /s /t 60")
