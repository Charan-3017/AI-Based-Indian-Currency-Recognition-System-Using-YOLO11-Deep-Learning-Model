from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")

model.train(
    data=r"C:\Users\cheta\OneDrive\Desktop\CURRENCY NEW\Currency Identification",
    epochs=50,
    imgsz=224,
    batch=16
)