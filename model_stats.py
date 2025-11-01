# =========================================================
# Auto summary Params & FLOPs for YOLO models (best.pt only, fixed names)
# =========================================================
from ultralytics import YOLO
from pathlib import Path
import torch
import csv
from thop import profile

ROOT = Path("runs_yolo")
OUT_CSV = "model_stats.csv"
IMGSZ = 640

# Liên kết thư mục → tên hiển thị
MODEL_NAMES = {
    "train_waid_cfg": "Base",
    "train_waid_p2": "Add P2",
    "train_waid_r_p5": "Remove P5",
    "train_waid_r_p5_ssff_dfe_cpam": "Remove P5 + SSFF/DFE/CPAM"
}

folders = [p for p in ROOT.iterdir() if p.is_dir()]
print(f"🔍 Tìm thấy {len(folders)} thư mục mô hình trong {ROOT}")

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Model Name", "Folder", "Parameters (M)", "FLOPs (GFLOPs)"])

    for folder in folders:
        # Lấy tên mô hình theo map, fallback về tên thư mục
        model_name = MODEL_NAMES.get(folder.name, folder.name)

        # Tìm best.pt trong thư mục chính hoặc /weights
        pt_path = None
        if (folder / "best.pt").exists():
            pt_path = folder / "best.pt"
        elif (folder / "weights" / "best.pt").exists():
            pt_path = folder / "weights" / "best.pt"

        if not pt_path:
            print(f"⚠️ Bỏ qua {folder.name} (không có best.pt)")
            continue

        try:
            model = YOLO(str(pt_path))
            m = model.model
            dummy = torch.zeros(1, 3, IMGSZ, IMGSZ)
            flops, params = profile(m, inputs=(dummy,), verbose=False)
            params_m = round(params / 1e6, 3)
            flops_g = round(flops / 1e9, 3)

            writer.writerow([model_name, folder.name, params_m, flops_g])
            print(f"✅ {model_name}: {params_m}M params | {flops_g} GFLOPs")

        except Exception as e:
            print(f"⚠️ Lỗi khi đọc {folder.name}: {e}")

print(f"\n📁 Xuất kết quả vào: {OUT_CSV}")
