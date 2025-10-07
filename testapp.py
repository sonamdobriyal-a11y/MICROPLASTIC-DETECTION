from ultralytics import YOLO

model = YOLO("runs/microplastics/weights/best.pt")
result = model.predict(
    source="C:/Users/Admin/Desktop/Sonam/microplastic-detection/data/LOW.jpg",  # replace with your image
    conf=0.3,
    save=True,           # writes annotated image to runs\detect\predict\
    device="cpu"         # or "cuda:0" if you have a GPU
)

print(result[0].boxes.xyxy)  # bounding boxes (optional)
