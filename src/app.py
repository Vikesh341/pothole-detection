import streamlit as st
import csv
import os
import sqlite3
import cv2
import database
from datetime import datetime
from ultralytics import YOLO

model = YOLO("runs/detect/train-2/weights/best.pt")
st.title("AI Road Damage Detection System")

st.write("Upload a road image to detect potholes and road cracks.")

uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)
location = st.text_input(
    "Enter Location",
    placeholder="e.g. patna, Bihar"
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Road Image")

    with open("temp_image.jpg", "wb") as file:
        file.write(uploaded_file.getbuffer())

    results = model("temp_image.jpg", conf=0.25)

    st.subheader("Detection Result")
    detected_data = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            detected_data.append({
            "detection": class_name,
            "confidence": confidence
        })

            st.write("Damage:", class_name)
            st.write("Confidence:", round(confidence, 2))
            st.write("Location:", location)
            
            image_height, image_width = result.orig_shape

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

            detected_data[-1]["area"] = area_percentage
            detected_data[-1]["severity"] = severity
        st.write("Area:", round(area_percentage, 2), "%")
        st.write("Estimated Severity:", severity)
        annotated_image = result.plot()
        os.makedirs("reports", exist_ok=True)
        annotated_image_path = f"reports/{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        cv2.imwrite(annotated_image_path, annotated_image)
        st.image(annotated_image, caption="Detected Road Damage")


if st.button("Generate Report"):
    report_id = datetime.now().strftime("%Y%m%d%H%M%S")
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    connection = sqlite3.connect("reports.db")
    cursor = connection.cursor()

    with open("reports.csv", "a", newline="") as file:
        writer = csv.writer(file)

        if os.path.getsize("reports.csv") == 0:
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

        for data in detected_data:
         writer.writerow([
         report_id,
         report_time,
         "Uploaded Image",
         annotated_image_path,
         data["detection"],
         round(data["confidence"], 2),
         round(data["area"], 2),
         data["severity"],
         location
    ])

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
        "Uploaded Image",
        annotated_image_path,
        data["detection"],
        data["confidence"],
        data["area"],
        data["severity"],
        location
    ))
    connection.commit()
    connection.close()

    st.success("Report generated successfully!")
    st.write("Report ID:", report_id)
    st.write("Date & Time:", report_time)
    st.header("Road Damage Dashboard")

connection = sqlite3.connect("reports.db")

dashboard_data = connection.execute("""
SELECT detection, severity, confidence, location, date_time, evidence_image, status
FROM reports
ORDER BY date_time DESC
""").fetchall()

connection.close()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reports", len(dashboard_data))

pothole_count = sum(1 for report in dashboard_data if report[0] == "pothole")
col2.metric("Potholes", pothole_count)

high_count = sum(1 for report in dashboard_data if report[1] == "High")
col3.metric("High Severity", high_count)
medium_count = sum(1 for report in dashboard_data if report[1] == "Medium")
low_count = sum(1 for report in dashboard_data if report[1] == "Low")

st.write("Medium Severity:", medium_count)
st.write("Low Severity:", low_count)

location_count = len(set(report[3] for report in dashboard_data))
col4.metric("Locations", location_count)
st.subheader("Recent Reports")

for report in dashboard_data:
    st.write({
        "Detection": report[0],
        "Severity": report[1],
        "Confidence": round(report[2], 2),
        "Location": report[3],
        "Date & Time": report[4]
    })
st.subheader("Recent Reports")
import pandas as pd



damage_counts = {}

for report in dashboard_data:
    damage = report[0]
    damage_counts[damage] = damage_counts.get(damage, 0) + 1

chart_data = pd.DataFrame(
    list(damage_counts.items()),
    columns=["Damage Type", "Reports"]
)


report_table = []

for report in dashboard_data:
    report_table.append({
    "Detection": report[0],
    "Severity": report[1],
    "Confidence": round(report[2], 2),
    "Location": report[3],
    "Date & Time": report[4],
    "Evidence Image": report[5],
    "Status": report[6]
})
st.dataframe(report_table, use_container_width=True)
st.subheader("Evidence Images")

for report in dashboard_data:
    evidence_path = report[5]

    if evidence_path and os.path.exists(evidence_path):
        st.image(
            evidence_path,
            caption=f"{report[0]} - {report[1]} - {report[3]}"
        )
        st.subheader("Update Report Status")

connection = sqlite3.connect("reports.db")

status_reports = connection.execute("""
SELECT report_id, detection, location, status
FROM reports
ORDER BY date_time DESC
""").fetchall()

connection.close()

if status_reports:
    selected_report = st.selectbox(
        "Select Report",
        status_reports,
        format_func=lambda x: f"{x[0]} - {x[1]} - {x[2]}"
    )

    new_status = st.selectbox(
        "New Status",
        ["Pending", "In Progress", "Resolved"]
    )

    if st.button("Update Status"):
        connection = sqlite3.connect("reports.db")

        connection.execute(
            "UPDATE reports SET status = ? WHERE report_id = ?",
            (new_status, selected_report[0])
        )

        connection.commit()
        connection.close()

        st.success("Status updated successfully!")
        st.rerun()