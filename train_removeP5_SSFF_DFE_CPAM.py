from ultralytics import YOLO
import torch
import os
import time
from pathlib import Path

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
print(torch.backends.cudnn.version())

DATA = "data.yaml"
PRETRAINED = "yolo12n.pt"
CFG = "models/yolov12n_removeP5_SSFF_DFE_CPAM.yaml"

if __name__ == "__main__":
    print("=== Bắt đầu huấn luyện YOLOv12 ===")
    model = YOLO(CFG).load(PRETRAINED)

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
        name="train_waid_r_p5_ssff_dfe_cpam",
        exist_ok=True,
        cache=True
    )

    # chờ file best.pt được ghi xong
    out_dir = Path(results.save_dir)
    best_model = out_dir / "weights" / "best.pt"
    while not best_model.exists():
        print("Chờ ghi file best.pt...")
        time.sleep(5)

    print("=== Huấn luyện và ghi log hoàn tất. Sẽ tắt máy sau 5 phút... ===")
    time.sleep(300)  # chờ 5 phút (300 giây)
    os.system("shutdown /s /t 60")
