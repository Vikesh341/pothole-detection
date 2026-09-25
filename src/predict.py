
from ultralytics import YOLO
import csv
import os
from datetime import datetime
import sqlite3
model = YOLO("runs/detect/train-2/weights/best.pt")

image_path = "src/road.jpg"
results = model(image_path, conf=0.25)
location = input("Enter location: ")
report_id = datetime.now().strftime("%Y%m%d%H%M%S")
report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

report_file = "reports.csv"

file_exists = os.path.exists(report_file)
connection = sqlite3.connect("reports.db")
cursor = connection.cursor()

with open(report_file, "a", newline="") as file:
    writer = csv.writer(file)

    if not file_exists:
        writer.writerow([
            "Report ID",
            "Date & Time",
    "Image",
    "Evidence Image",
    "Detection",
    "Confidence",
    "Area Percentage",
    "Estimated Severity",
    "Location"
])
       

    for result in results:
        image_height, image_width = result.orig_shape

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            box_width = x2 - x1
            box_height = y2 - y1

            box_area = box_width * box_height
            image_area = image_width * image_height

            area_percentage = (box_area / image_area) * 100

            if area_percentage < 5:
                severity = "Low"
            elif area_percentage < 15:
                severity = "Medium"
            else:
                severity = "High"

            print("Detected:", class_name)
            print("Confidence:", round(confidence, 2))
            print("Area:", round(area_percentage, 2), "%")
            print("Estimated Severity:", severity)
            cursor.execute("""
INSERT INTO reports (
    report_id,
    date_time,
    image,
    evidence_image,
    detection,
    confidence,
    area_percentage,
    severity,
    location
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    report_id,
    report_time,
    image_path,
    f"reports/{report_id}.jpg",
    class_name,
    confidence,
    area_percentage,
    severity,
    location
))

            writer.writerow([
            report_id,
            report_time,
            image_path,
            f"reports/{report_id}.jpg",
            class_name,
            round(confidence, 2),
            round(area_percentage, 2),
            severity,
            location
            ])
result = results[0]
os.makedirs("reports", exist_ok=True)

result.save(filename=f"reports/{report_id}.jpg")
print(f"Annotated image saved to: reports/{report_id}.jpg")
print("Report saved to:", report_file)
connection.commit()
connection.close()

print("Report saved to database!")