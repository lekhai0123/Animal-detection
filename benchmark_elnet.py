import os, sys, time, csv, json
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from torchinfo import summary
from thop import profile
import torch

# =========================
# DANH SÁCH YAML CẦN TEST
# =========================
MODELS = [
    "models/elnet/elnet12n_base.yaml",
    "models/elnet/elnet12n_base_edown.yaml",
    "models/elnet/elnet12n_base_edown_dimb.yaml",
    "models/elnet/elnet12n_full.yaml",
]
DATA = "data.yaml"

# =========================
# LOG
# =========================
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = log_dir / f"benchmark_elnet_{timestamp}.log"

class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self): [f.flush() for f in self.files]

log_file = open(log_path, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

# =========================
# BẢNG KẾT QUẢ
# =========================
csv_path = Path("benchmark_elnet_results.csv")
csv_header = ["model", "params(M)", "flops(G)", "fps", "precision", "recall", "mAP50", "mAP50_95"]
if not csv_path.exists():
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(csv_header)

def measure_flops_params(model):
    x = torch.randn(1, 3, 640, 640).cuda() if torch.cuda.is_available() else torch.randn(1, 3, 640, 640)
    flops, params = profile(model, inputs=(x,), verbose=False)
    return round(params / 1e6, 3), round(flops / 1e9, 3)

def measure_fps(model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(8, 3, 640, 640).to(device)
    model.to(device).eval()
    for _ in range(10): model(x)  # warmup
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    for _ in range(50): model(x)
    torch.cuda.synchronize() if device == "cuda" else None
    fps = 50 * 8 / (time.time() - t0)
    return round(fps, 2)

# =========================
# CHẠY TOÀN BỘ
# =========================
def main():
    results_list = []
    for model_path in MODELS:
        print(f"\n=== 🔹 Huấn luyện {model_path} ===")
        name = Path(model_path).stem
        try:
            model = YOLO(model_path)
            results = model.train(
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
                name=f"train_{name}",
                exist_ok=True,
                cache=True
            )

            # --- Đánh giá ---
            print(f"Đang đánh giá {name} ...")
            metrics = model.val(data=DATA, split="val")
            precision = float(metrics.results_dict.get("metrics/precision(B)", 0))
            recall = float(metrics.results_dict.get("metrics/recall(B)", 0))
            map50 = float(metrics.results_dict.get("metrics/mAP50(B)", 0))
            map5095 = float(metrics.results_dict.get("metrics/mAP50-95(B)", 0))

            # --- FLOPs, Params, FPS ---
            params, flops = measure_flops_params(model.model)
            fps = measure_fps(model.model)

            results_list.append([name, params, flops, fps, precision, recall, map50, map5095])
            print(f"✅ {name}: Params={params}M, FLOPs={flops}G, FPS={fps}, P={precision:.3f}, R={recall:.3f}, mAP50={map50:.3f}, mAP50-95={map5095:.3f}")

            # --- Ghi CSV ---
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([name, params, flops, fps, precision, recall, map50, map5095])

        except Exception as e:
            print(f"❌ Lỗi khi huấn luyện {name}: {e}")
            continue

    # =========================
    # TỔNG KẾT
    # =========================
    print("\n=== 📊 KẾT QUẢ CUỐI CÙNG ===")
    for r in results_list:
        print(f"{r[0]:25s} | Params {r[1]:5.2f}M | FLOPs {r[2]:5.2f}G | FPS {r[3]:5.1f} | P {r[4]:.3f} | R {r[5]:.3f} | mAP50 {r[6]:.3f} | mAP50-95 {r[7]:.3f}")

    print(f"\nBảng kết quả đã lưu: {csv_path}")
    print("=== Hoàn tất benchmark. Sẽ tắt máy sau 10 phút ===")
    time.sleep(600)
    os.system("shutdown /s /t 0")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()