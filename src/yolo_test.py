from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("src/road.jpg")

results[0].show()