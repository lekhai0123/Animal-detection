import json, re

src = "WAID/annotations/test.json"
dst = "WAID/annotations/test_clean.json"

with open(src) as f:
    data = json.load(f)

# Giữ lại chỉ ảnh có tên chứa '_jpg'
keep_images = [img for img in data["images"] if re.search(r"_jpg", img["file_name"])]
keep_ids = {img["id"] for img in keep_images}
keep_annotations = [ann for ann in data["annotations"] if ann["image_id"] in keep_ids]

data["images"] = keep_images
data["annotations"] = keep_annotations

with open(dst, "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Đã tạo {dst} với {len(keep_images)} ảnh và {len(keep_annotations)} annotations.")
