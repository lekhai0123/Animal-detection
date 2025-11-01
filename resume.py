from ultralytics import YOLO
from multiprocessing import freeze_support
from pathlib import Path

def main():
    ckpt = "runs_yolo/train_elnet12n_base_edown_dimb/weights/last.pt"
    model = YOLO(ckpt if Path(ckpt).exists() else "models/elnet/elnet12n_base_edown_dimb.yaml")
    model.train(data="data.yaml", resume=Path(ckpt).exists(), save_period=1)

if __name__ == "__main__":
    freeze_support()
    main()
