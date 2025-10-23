from ultralytics import YOLO
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)
print(torch.backends.cudnn.version())

DATA = "data.yaml"
PRETRAINED = "yolo12n.pt"
CFG = "models/yolov12n_p2.yaml"

if __name__ == "__main__":
    # tạo model mới từ YAML, khởi tạo từ trọng số pretrain
    model = YOLO(CFG).load(PRETRAINED)

    model.train(
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
        name="train_waid_p2",
        exist_ok=True,
        cache=True
    )
