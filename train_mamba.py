from ultralytics import YOLO
import torch
import os
import time
from pathlib import Path
from datetime import datetime
import sys

# ======== GHI LOG SONG SONG RA FILE ========
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = log_dir / f"train_{timestamp}.log"

class Tee:
    """Ghi log ra cả terminal và file"""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open(log_path, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

# ======== BẮT ĐẦU SCRIPT GỐC ========
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
print(torch.backends.cudnn.version())

DATA = "data.yaml"
PRETRAINED = "yolo12n.pt"
CFG = "models/yolov12n_mamba.yaml"

if __name__ == "__main__":
    print("=== Bắt đầu huấn luyện YOLOv12 ===")
    model = YOLO(CFG).to('cuda')
    model.model.model[-1].ch = [256, 512, 1024]

    results = model.train(
        data=DATA,
        imgsz=640,
        epochs=100,
        batch=16,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        device=0,
        workers=0,
        project="runs_yolo",
        name="train_waid_mamba",
        exist_ok=True,
        cache=True
    )

    # chờ file best.pt được ghi xong
    out_dir = Path(results.save_dir)
    best_model = out_dir / "weights" / "best.pt"
    while not best_model.exists():
        print("Chờ ghi file best.pt...")
        time.sleep(5)

    print(f"=== Huấn luyện và ghi log hoàn tất. Log được lưu tại: {log_path} ===")
    print("=== Sẽ tắt máy sau 5 phút... ===")
    time.sleep(300)  # chờ 5 phút (300 giây)
    os.system("shutdown /s /t 60")
