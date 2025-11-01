import os, shutil

# Đường dẫn gốc
root = r"WAID/images"
src = os.path.join(root, "test_invalid")
dst = os.path.join(root, "test")

moved = 0
skipped = 0

if not os.path.exists(src):
    print(f"❌ Không tìm thấy thư mục nguồn: {src}")
    exit()

if not os.path.exists(dst):
    os.makedirs(dst, exist_ok=True)

# Di chuyển tất cả ảnh từ test_invalid về test
for f in os.listdir(src):
    if not f.lower().endswith((".jpg", ".png")):
        skipped += 1
        continue
    shutil.move(os.path.join(src, f), os.path.join(dst, f))
    moved += 1

print(f"✅ Đã di chuyển {moved} ảnh từ {src} về {dst}")
if skipped:
    print(f"⚠️ Bỏ qua {skipped} file không phải ảnh.")
