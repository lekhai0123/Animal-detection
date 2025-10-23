from pathlib import Path
from ultralytics import YOLO
import glob, os, csv
import numpy as np
from tqdm import tqdm
import cv2

# === cấu hình ===
DATA = "data.yaml"
IMG_DIR = "WAID/images/test"
LABEL_DIR = "WAID/labels/test"
MODELS = [
    "runs_yolo/train_waid_cfg/weights/best.pt",
    "runs_yolo/train_waid_p2/weights/best.pt",
]
THRESH_PX = 50
IMG_SIZE = 640
IOU_THRESH = 0.5
OUT_CSV = "results_small_objects.csv"

# === hàm tiện ích ===
def load_labels(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f.readlines():
            c, x, y, w, h = map(float, line.strip().split())
            bw, bh = w * img_w, h * img_h
            if bw <= THRESH_PX and bh <= THRESH_PX:
                boxes.append([x * img_w, y * img_h, bw, bh, int(c)])
    return boxes

def box_iou(box1, box2):
    b1 = np.array([box1[0]-box1[2]/2, box1[1]-box1[3]/2, box1[0]+box1[2]/2, box1[1]+box1[3]/2])
    b2 = np.array([box2[0]-box2[2]/2, box2[1]-box2[3]/2, box2[0]+box2[2]/2, box2[1]+box2[3]/2])
    inter_x1, inter_y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    inter_x2, inter_y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def evaluate(model_path):
    model = YOLO(model_path)
    # lấy tên thư mục chứa model (vd: train_waid_cfg)
    model_name = Path(model_path).parts[-3] if "weights" in model_path else Path(model_path).stem

    imgs = sorted(glob.glob(os.path.join(IMG_DIR, "*.*")))
    TP, FP, FN = 0, 0, 0

    for img_path in tqdm(imgs, desc=f"Evaluating {model_name}"):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        gt = load_labels(os.path.join(LABEL_DIR, os.path.basename(img_path).replace(".jpg", ".txt")), w, h)
        if len(gt) == 0:
            continue

        preds = model.predict(source=img_path, imgsz=IMG_SIZE, conf=0.25, iou=0.6, verbose=False)[0]
        dets = []
        for box in preds.boxes.xywh.cpu().numpy():
            bw, bh = box[2], box[3]
            if bw <= THRESH_PX and bh <= THRESH_PX:
                dets.append(box)

        matched = set()
        for d in dets:
            if any(box_iou(d, g) >= IOU_THRESH for g in gt):
                TP += 1
                matched.add(tuple(d))
            else:
                FP += 1
        FN += max(0, len(gt) - len(matched))

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {
        "Model": model_name,  # tên thư mục thay vì best.pt
        "Small_Object_Threshold(px)": THRESH_PX,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "TP": TP, "FP": FP, "FN": FN
    }


if __name__ == "__main__":
    results = []
    for m in MODELS:
        r = evaluate(m)
        print(r)
        results.append(r)

    # === lưu csv ===
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to {OUT_CSV}")
