
from ultralytics import YOLO
import multiprocessing

def main():
    model = YOLO("runs_yolo/train_waid_r_p5/weights/best.pt")
    metrics = model.val(
        data="data.yaml",
        split="test",      # tập test
        imgsz=640,
        batch=8,
        device=0,
        workers=3          # nếu vẫn lỗi → đổi thành 0
    )
    print(metrics)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
