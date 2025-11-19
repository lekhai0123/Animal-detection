from ultralytics import YOLO
import os
import shutil
import pandas as pd

MODEL_PATH = "runs_yolo/train_waid_cfg/weights/best.pt"
DATA_YAML_PATH = "data.yaml"
ORIGINAL_TEST_DIR = "WAID/images/test"
FILTERED_DEST_DIR = "WAID/test/filtered_images"
TARGET_CLASSES = {3, 4, 5}

def run_validation(model, data_yaml_path):
    metrics = model.val(data=data_yaml_path, split='test', verbose=False, workers=0)
    precision = metrics.box.p[0]
    recall = metrics.box.r[0]
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    return {
        "Precision": precision,
        "Recall": recall,
        "mAP@50": map50,
        "mAP@50-95": map50_95,
    }

def filter_images(model, source_dir, dest_dir, target_classes):
    os.makedirs(dest_dir, exist_ok=True)
    results = model(source_dir, save=False, verbose=False, stream=True)
    moved_files = []
    for r in results:
        if any(int(cls) in target_classes for cls in r.boxes.cls.cpu().numpy()):
            filename = os.path.basename(r.path)
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            if os.path.exists(source_path):
                shutil.move(source_path, dest_path)
                moved_files.append(filename)
    return moved_files

def restore_images(source_dir, dest_dir):
    if not os.path.exists(source_dir):
        return
    files_to_move = os.listdir(source_dir)
    for filename in files_to_move:
        shutil.move(os.path.join(source_dir, filename), os.path.join(dest_dir, filename))
    if len(os.listdir(source_dir)) == 0:
        os.rmdir(source_dir)

def main():
    model = YOLO(MODEL_PATH)
    restore_images(FILTERED_DEST_DIR, ORIGINAL_TEST_DIR)

    try:
        results_original = run_validation(model, DATA_YAML_PATH)
        filter_images(model, ORIGINAL_TEST_DIR, FILTERED_DEST_DIR, TARGET_CLASSES)
        results_filtered = run_validation(model, DATA_YAML_PATH)

        comparison_data = {
            "Tập Test Gốc": results_original,
            "Tập Test Đã Lọc (bỏ lớp 3,4,5)": results_filtered
        }
        df = pd.DataFrame(comparison_data)
        print(df.to_string())

        model.benchmark()

    finally:
        restore_images(FILTERED_DEST_DIR, ORIGINAL_TEST_DIR)

if __name__ == '__main__':
    main()