if __name__ == "__main__":
    from ultralytics import YOLO
    from pathlib import Path

    DATA = "data.yaml"
    RUNS_DIR = Path("runs_yolo")

    model_paths = list(RUNS_DIR.glob("**/weights/best.pt"))
    print(f"🔍 Found {len(model_paths)} models to re-evaluate")

    for m_path in model_paths:
        model_name = Path(m_path).parent.parent.name
        print(f"\n🔹 Evaluating {model_name} ...")
        model = YOLO(m_path)
        model.val(
            data=DATA,
            split="test",
            save_json=True,
            name=f"val_{model_name}"  # tạo thư mục riêng
        )
