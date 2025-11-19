import os, json, yaml
from tqdm import tqdm
from PIL import Image

with open("data.yaml","r") as f:
    cfg = yaml.safe_load(f)

base_path = cfg["path"]
images_dir = os.path.join(base_path,"images/test")
labels_dir = os.path.join(base_path,"labels/test")
output_json = os.path.join(base_path,"annotations/test.json")
classes = cfg["names"]

def yolo_to_coco_bbox(b, w, h):
    x,y,ww,hh = map(float,b)
    x = (x - ww/2)*w
    y = (y - hh/2)*h
    return [x, y, ww*w, hh*h]

images, annotations = [], []
ann_id, img_id = 1, 0
valid_ext = (".jpg",".jpeg",".png",".bmp",".webp")

for file in tqdm(sorted(os.listdir(images_dir))):
    if not file.lower().endswith(valid_ext):
        continue
    img_path = os.path.join(images_dir,file)
    label_path = os.path.join(labels_dir, os.path.splitext(file)[0]+".txt")

    with Image.open(img_path) as im:
        w,h = im.size

    images.append({"id": img_id, "width": w, "height": h, "file_name": file})

    if os.path.exists(label_path):
        with open(label_path,"r") as f:
            for line in f:
                p = line.strip().split()
                if len(p) < 5:
                    continue
                c = int(float(p[0]))
                bx = list(map(float,p[1:5]))
                bb = yolo_to_coco_bbox(bx,w,h)
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": c,
                    "bbox": bb,
                    "area": bb[2]*bb[3],
                    "iscrowd": 0,
                    "segmentation": []
                })
                ann_id += 1
    img_id += 1

categories = [{"id": i, "name": name} for i,name in enumerate(classes)]

coco = {
    "info": {"description":"WAID test set in COCO format","version":"1.0"},
    "licenses": [],
    "images": images,
    "annotations": annotations,
    "categories": categories
}

os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json,"w") as f:
    json.dump(coco,f,indent=2)

print(f"Tạo {output_json}")
print(f"images={len(images)}, anns={len(annotations)}, classes={len(classes)}")
