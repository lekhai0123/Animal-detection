from ultralytics import YOLO

MODEL = "yolo12n.pt"   
DATA  = "data.yaml"     

if __name__ == "__main__":
    model = YOLO(MODEL)
    model.train(
        data=DATA,
        imgsz=640,         
        epochs=100,           
        batch=4,           
        optimizer="SGD",     
        lr0=0.01,             
        lrf=0.01,             
        momentum=0.937,       
        weight_decay=0.0005,  
        device=0,             
        workers=0,
        project="runs_yolo",
        name="train_waid_cfg",
        exist_ok=True,
        amp=False,
    )
