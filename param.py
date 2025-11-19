from ultralytics import YOLO
from pathlib import Path
import torch
import csv
from thop import profile

RUNS_DIR = Path("runs_yolo")
OUT_CSV = "model_params.csv"
INPUT_SIZE = (1, 3, 640, 640)  # batch=1, 3 kênh, 640x640

def count_params_and_flops(model_path: Path):
    try:
        model = YOLO(str(model_path))
        model.model.eval()
        dummy_input = torch.randn(INPUT_SIZE)
        flops, params = profile(model.model, inputs=(dummy_input,), verbose=False)
        return round(params / 1e6, 3), round(flops / 1e9, 3)  # FLOPs tính theo GFLOPs
    except Exception as e:
        print(f"[ERROR] {model_path}: {e}")
        return None, None

def main():
    rows = []
    for pt_file in RUNS_DIR.glob("**/weights/best.pt"):
        model_name = pt_file.parts[-3]
        params_m, flops_g = count_params_and_flops(pt_file)
        if params_m is not None:
            print(f"{model_name}: {params_m}M params | {flops_g} GFLOPs")
            rows.append({
                "Model": model_name,
                "Path": str(pt_file),
                "Params(M)": params_m,
                "FLOPs(G)": flops_g
            })

    if rows:
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved summary to {OUT_CSV}")
    else:
        print("No models found.")

if __name__ == "__main__":
    main()
