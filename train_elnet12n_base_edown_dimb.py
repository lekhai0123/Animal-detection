from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import sys, os, time, multiprocessing


MODEL = "models/elnet/elnet12n_base_edown_dimb.yaml"
DATA = "data.yaml"


def main():
    # === Ghi log song song ra terminal và file ===
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{Path(MODEL).stem}_{timestamp}.log"

    class Tee:
        def __init__(self, *files): self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            [f.flush() for f in self.files]

    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print(f"=== Bắt đầu huấn luyện {MODEL} ===")
    model = YOLO(MODEL)
    results = model.train(
        data=DATA,
        imgsz=640,
        epochs=100,
        batch=8,
        optimizer="AdamW",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        device=0,
        workers=0, 
        project="runs_yolo",
        name=f"train_{Path(MODEL).stem}",
        exist_ok=True,
        cache="disk",
        save_period=1,
    )
    
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
