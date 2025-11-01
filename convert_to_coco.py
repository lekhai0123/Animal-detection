import os, json, yaml, cv2
from pathlib import Path

DATA_YAML = "data.yaml"
SPLIT = "test"
OUT_JSON = "WAID/annotations/test.json"

with open(DATA_YAML, "r") as f:
    data = yaml.safe_load(f)

root = Path(data.get("path", "")).resolve()
img_dir = root / data.get(SPLIT, f"images/{SPLIT}")
label_dir = root / f"labels/{SPLIT}"

assert img_dir.exists(), f"Missing images: {img_dir}"
assert label_dir.exists(), f"Missing labels: {label_dir}"

names = data.get("names") or data.get("classes")
if isinstance(names, dict):
    categories = [{"id": int(k)+1, "name": v} for k, v in names.items()]
else:
    categories = [{"id": i+1, "name": n} for i, n in enumerate(names)]

images, annotations = [], []
aid = 1
img_files = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])

for iid, img_path in enumerate(img_files, start=1):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]
    # stem chuẩn: bỏ hậu tố .rf nếu có
    # giữ nguyên tên file (bao gồm cả phần .rf nếu có)
    images.append({
        "id": iid,
        "width": w,
        "height": h,
        "file_name": img_path.stem + img_path.suffix  # vd: camframe00000540.jpg
    })


    lab = label_dir / (img_path.stem + ".txt")
    if not lab.exists():
        continue
    for line in open(lab):
        p = line.strip().split()
        if len(p) != 5:
            continue
        c, x, y, bw, bh = map(float, p)
        x, y, bw, bh = x*w, y*h, bw*w, bh*h
        x1, y1 = x - bw/2, y - bh/2
        annotations.append({
            "id": aid,
            "image_id": iid,
            "category_id": int(c)+1,
            "bbox": [x1, y1, bw, bh],
            "area": bw*bh,
            "iscrowd": 0
        })
        aid += 1

os.makedirs(Path(OUT_JSON).parent, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump({
        "info": {"description": "WAID test COCO", "version":"1.0"},
        "images": images,
        "annotations": annotations,
        "categories": categories
    }, f)

print("Images:", len(images))
print("Anns:", len(annotations))
