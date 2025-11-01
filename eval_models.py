# =========================================================
# eval_small_summary_all.py
# Đánh giá toàn bộ mô hình YOLOv12 trong runs/detect/
# Chỉ tính metrics trên đối tượng nhỏ hơn 50px
# Xuất:
#   1. metrics_small_summary.csv  (chi tiết từng model)
#   2. metrics_small_overall.csv  (trung bình 4 model)
# =========================================================
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path
import json, csv, os, sys

# ==== CẤU HÌNH ====
GT_JSON   = "WAID/annotations/test.json"   # Ground truth COCO
ROOT_DIR  = Path("runs/detect")            # Thư mục chứa model
THRESH_PX = 50                             # Giới hạn kích thước nhỏ (px)
OUT_DETAIL = "metrics_small_summary.csv"   # Kết quả từng model
OUT_MEAN   = "metrics_small_overall.csv"   # Kết quả trung bình

# ==== HÀM PHỤ ====
def bbox_is_small(bbox, thresh=50):
    w, h = bbox[2], bbox[3]
    return max(w, h) < thresh

def normalize_name(name: str):
    base = os.path.basename(name)
    root, _ = os.path.splitext(base)
    return root.lower()

def evaluate_small(pred_json_path: Path, coco_gt: COCO, small_img_ids: list[int]):
    """Đánh giá mAP, Precision, Recall cho small objects"""
    with open(pred_json_path, "r") as f:
        preds = json.load(f)

    # map file_name → image_id từ GT
    name_to_id = {normalize_name(img["file_name"]): img["id"]
                  for img in coco_gt.dataset["images"]}

    # lọc prediction trùng ID nhỏ
    small_preds, miss_count = [], 0
    for p in preds:
        img_name_raw = str(p.get("image_id"))
        norm = normalize_name(img_name_raw)
        if norm in name_to_id:
            coco_id = name_to_id[norm]
            if coco_id in small_img_ids:
                p["image_id"] = coco_id
                small_preds.append(p)
        else:
            miss_count += 1

    if len(small_preds) == 0:
        print(f"⚠️ {pred_json_path.parent.name}: không có prediction hợp lệ (<{THRESH_PX}px)")
        return None

    tmp_json = pred_json_path.parent / "predictions_small_tmp.json"
    with open(tmp_json, "w") as f:
        json.dump(small_preds, f)

    # đánh giá
    coco_dt = coco_gt.loadRes(str(tmp_json))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = small_img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # lấy các chỉ số small
    (
        mAP_all, mAP50, mAP75,
        AP_small, AP_medium, AP_large,
        AR_all, AR50, AR75,
        AR_small, AR_medium, AR_large
    ) = coco_eval.stats

    precision = AP_small
    recall = AR_small
    mAP = mAP_all

    return precision, recall, mAP


# ==== BẮT ĐẦU ====
print(f"📂 Ground truth: {GT_JSON}")
if not os.path.exists(GT_JSON):
    print("❌ Không tìm thấy file ground truth.")
    sys.exit()

coco_gt = COCO(GT_JSON)

# lọc ảnh có vật nhỏ
small_img_ids = []
for img_id, anns in coco_gt.imgToAnns.items():
    for ann in anns:
        if bbox_is_small(ann["bbox"], THRESH_PX):
            small_img_ids.append(img_id)
            break
print(f"🔹 Ảnh có vật nhỏ (<{THRESH_PX}px): {len(small_img_ids)}")

# tìm predictions.json
folders = [p for p in ROOT_DIR.iterdir() if p.is_dir()]
print(f"🔍 Phát hiện {len(folders)} mô hình trong {ROOT_DIR}")

results = []

# ---- ĐÁNH GIÁ TỪNG MODEL ----
with open(OUT_DETAIL, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model Name", "Folder Path", "Precision (AP_small)", "Recall (AR_small)", "mAP_small (COCO)"])

    for folder in folders:
        pred_json = folder / "predictions.json"
        if not pred_json.exists():
            print(f"⏭️ Bỏ qua {folder.name} (không có predictions.json)")
            continue

        print(f"\n🚀 Đang đánh giá model: {folder.name}")
        result = evaluate_small(pred_json, coco_gt, small_img_ids)
        if result is None:
            continue

        precision, recall, mAP = result
        writer.writerow([
            folder.name,
            str(folder),
            round(precision, 4),
            round(recall, 4),
            round(mAP, 4)
        ])
        results.append((precision, recall, mAP))

print(f"\n✅ Hoàn tất. File chi tiết: {OUT_DETAIL}")

# ---- TÍNH TRUNG BÌNH ----
if results:
    avg_precision = sum(r[0] for r in results) / len(results)
    avg_recall = sum(r[1] for r in results) / len(results)
    avg_map = sum(r[2] for r in results) / len(results)

    with open(OUT_MEAN, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Avg Precision (AP_small)", "Avg Recall (AR_small)", "Avg mAP_small (COCO)", "Num Models"])
        writer.writerow([
            round(avg_precision, 4),
            round(avg_recall, 4),
            round(avg_map, 4),
            len(results)
        ])

    print(f"📊 Trung bình 4 mô hình: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, mAP={avg_map:.4f}")
    print(f"✅ Đã lưu file trung bình tại: {OUT_MEAN}")
else:
    print("❌ Không có mô hình hợp lệ để tính trung bình.")
