from ultralytics import YOLO
from pathlib import Path
import csv

# === Cấu hình ===
RUNS_DIR = Path("runs_yolo")
OUT_CSV = "model_params.csv"

def count_params(model_path: Path):
    try:
        model = YOLO(str(model_path))
        total = sum(p.numel() for p in model.model.parameters())
        return round(total / 1e6, 3)
    except Exception as e:
        print(f"[ERROR] {model_path}: {e}")
        return None

def main():
    rows = []
    for pt_file in RUNS_DIR.glob("**/weights/best.pt"):
        model_name = pt_file.parts[-3]  # vd: train_waid_p2
        total = count_params(pt_file)
        if total is not None:
            print(f"{model_name}: {total}M parameters")
            rows.append({
                "Model": model_name,
                "Path": str(pt_file),
                "Total_Params(M)": total
            })

    if rows:
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved summary to {OUT_CSV}")
    else:
        print("No models found in runs_yolo/**/weights/best.pt")

if __name__ == "__main__":
    main()
