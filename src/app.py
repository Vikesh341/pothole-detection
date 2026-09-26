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

import pandas as pd

# ==============================
# PROFESSIONAL DASHBOARD
# ==============================

# ==============================
# PROFESSIONAL DASHBOARD
# ==============================

# ---------- CUSTOM UI ----------

st.markdown("""
<style>

.main-title {
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 4px;
}

.main-subtitle {
    color: #8b949e;
    font-size: 15px;
    margin-bottom: 25px;
}

.metric-card {
    background: linear-gradient(145deg, #161b22, #0d1117);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: #8b949e;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    font-size: 32px;
    font-weight: 800;
    margin-top: 8px;
}

.section-header {
    font-size: 22px;
    font-weight: 750;
    margin-top: 30px;
    margin-bottom: 15px;
}

.report-card {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 15px;
}

.report-title {
    font-size: 19px;
    font-weight: 750;
    margin-bottom: 8px;
}

.report-info {
    color: #8b949e;
    font-size: 14px;
    line-height: 1.8;
}

.status-pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------

st.markdown(
    '<div class="main-title">🚧 Road Damage Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'AI-powered road damage detection, evidence tracking and report management'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# AI ROAD MONITORING CONTROL CENTER
# ==========================================================

# ---------- DASHBOARD HEADER ----------

st.title("🚧 AI Road Monitoring Center")

st.caption(
    "Intelligent road-damage detection • Evidence monitoring • Report management"
)

# System status
if dashboard_data:
    st.success("● AI SYSTEM ONLINE  |  Monitoring active road reports")
else:
    st.info("● AI SYSTEM READY  |  Waiting for road-damage detection")


# ==========================================================
# KPI OVERVIEW
# ==========================================================

total_reports = len(dashboard_data)

pothole_count = sum(
    1 for report in dashboard_data
    if report[0].lower() == "pothole"
)

high_count = sum(
    1 for report in dashboard_data
    if report[1] == "High"
)

medium_count = sum(
    1 for report in dashboard_data
    if report[1] == "Medium"
)

low_count = sum(
    1 for report in dashboard_data
    if report[1] == "Low"
)

location_count = len(
    set(report[3] for report in dashboard_data)
)


kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Total Reports",
        total_reports
    )

with kpi2:
    st.metric(
        "Potholes Detected",
        pothole_count
    )

with kpi3:
    st.metric(
        "High Risk",
        high_count
    )

with kpi4:
    st.metric(
        "Locations",
        location_count
    )


st.divider()


# ==========================================================
# LIVE DETECTION PANEL
# ==========================================================

st.subheader("📡 Latest AI Detection")

if dashboard_data:

    latest = dashboard_data[0]

    detection = latest[0]
    severity = latest[1]
    confidence = latest[2]
    location = latest[3]
    date_time = latest[4]
    evidence_path = latest[5]
    status = latest[6]

    camera_col, detection_col = st.columns(
        [1.7, 1]
    )

    # ---------- CAMERA ----------
    with camera_col:

        st.markdown("### 📷 Road Camera")

        if evidence_path and os.path.exists(evidence_path):

            st.image(
                evidence_path,
                caption="Latest AI-detected road condition",
                use_container_width=True
            )

        else:

            st.info(
                "No camera evidence available."
            )


    # ---------- AI RESULT ----------
    with detection_col:

        st.markdown("### 🤖 AI Analysis")

        st.metric(
            "Detected Damage",
            detection.title()
        )

        st.metric(
            "AI Confidence",
            f"{confidence * 100:.1f}%"
        )

        st.metric(
            "Estimated Severity",
            severity
        )

        st.metric(
            "Current Status",
            status
        )

        st.caption(
            f"📍 Location: {location}"
        )

        st.caption(
            f"🕒 Detected: {date_time}"
        )

else:

    st.info(
        "No road-damage detection available yet."
    )


st.divider()


# ==========================================================
# ANALYTICS CENTER
# ==========================================================

st.subheader("📊 Road Damage Analytics")

analytics1, analytics2 = st.columns(2)


# ---------- DAMAGE TYPE ----------

with analytics1:

    st.markdown("### Damage Distribution")

    damage_counts = {}

    for report in dashboard_data:

        damage = report[0]

        damage_counts[damage] = (
            damage_counts.get(damage, 0) + 1
        )

    if damage_counts:

        damage_chart = pd.DataFrame(
            list(damage_counts.items()),
            columns=[
                "Damage Type",
                "Reports"
            ]
        )

        st.bar_chart(
            damage_chart.set_index("Damage Type"),
            use_container_width=True
        )

    else:

        st.info("No damage data available.")


# ---------- SEVERITY ----------

with analytics2:

    st.markdown("### Severity Distribution")

    severity_data = pd.DataFrame(
        {
            "Severity": [
                "High",
                "Medium",
                "Low"
            ],
            "Reports": [
                high_count,
                medium_count,
                low_count
            ]
        }
    )

    severity_data = severity_data[
        severity_data["Reports"] > 0
    ]

    if not severity_data.empty:

        st.bar_chart(
            severity_data.set_index("Severity"),
            use_container_width=True
        )

    else:

        st.info(
            "No severity data available."
        )


# ==========================================================
# MONITORING SUMMARY
# ==========================================================

st.markdown("### 📈 Monitoring Summary")

summary1, summary2, summary3 = st.columns(3)

with summary1:

    if total_reports > 0:

        high_percentage = (
            high_count / total_reports
        ) * 100

    else:

        high_percentage = 0

    st.metric(
        "High-Risk Share",
        f"{high_percentage:.1f}%"
    )


with summary2:

    if total_reports > 0:

        avg_confidence = (
            sum(
                report[2]
                for report in dashboard_data
            )
            / total_reports
        ) * 100

    else:

        avg_confidence = 0

    st.metric(
        "Average AI Confidence",
        f"{avg_confidence:.1f}%"
    )


with summary3:

    resolved_count = sum(
        1
        for report in dashboard_data
        if report[6] == "Resolved"
    )

    st.metric(
        "Resolved Reports",
        resolved_count
    )


st.divider()


# ==========================================================
# RECENT DETECTIONS
# ==========================================================

st.subheader("🛰️ Recent Detection Feed")

if dashboard_data:

    for report in dashboard_data[:10]:

        detection = report[0]
        severity = report[1]
        confidence = report[2]
        location = report[3]
        date_time = report[4]
        status = report[6]

        with st.container(border=True):

            feed_col1, feed_col2, feed_col3 = st.columns(
                [2.5, 2, 1.2]
            )

            with feed_col1:

                st.markdown(
                    f"### 🚧 {detection.title()}"
                )

                st.caption(
                    f"📍 {location}"
                )

            with feed_col2:

                st.write(
                    f"**Severity:** {severity}"
                )

                st.write(
                    f"**Confidence:** "
                    f"{confidence * 100:.1f}%"
                )

                st.caption(
                    date_time
                )

            with feed_col3:

                if status == "Resolved":

                    st.success(
                        "Resolved"
                    )

                elif status == "In Progress":

                    st.warning(
                        "In Progress"
                    )

                else:

                    st.info(
                        "Pending"
                    )

else:

    st.info(
        "Detection feed is empty."
    )


st.divider()


# ==========================================================
# EVIDENCE GALLERY
# ==========================================================

st.subheader("🖼️ Detection Evidence")

evidence_reports = []

for report in dashboard_data:

    evidence_path = report[5]

    if (
        evidence_path
        and os.path.exists(evidence_path)
    ):

        evidence_reports.append(
            report
        )


if evidence_reports:

    image_columns = st.columns(3)

    for index, report in enumerate(
        evidence_reports[:9]
    ):

        evidence_path = report[5]

        with image_columns[index % 3]:

            st.image(
                evidence_path,
                caption=(
                    f"{report[0].title()} • "
                    f"{report[1]}"
                ),
                use_container_width=True
            )

            st.caption(
                f"📍 {report[3]}"
            )

else:

    st.info(
        "No detection evidence available."
    )


st.divider()


# ==========================================================
# REPORT MANAGEMENT
# ==========================================================

st.subheader("🔄 Report Management")

connection = sqlite3.connect(
    "reports.db"
)

status_reports = connection.execute(
    """
    SELECT
        report_id,
        detection,
        location,
        status
    FROM reports
    ORDER BY date_time DESC
    """
).fetchall()

connection.close()


if status_reports:

    management_col1, management_col2 = st.columns(
        [2, 1]
    )

    with management_col1:

        selected_report = st.selectbox(
            "Select Report",
            status_reports,
            format_func=lambda x:
                f"{x[0]} • "
                f"{x[1].title()} • "
                f"{x[2]}"
        )

    with management_col2:

        new_status = st.selectbox(
            "Change Status",
            [
                "Pending",
                "In Progress",
                "Resolved"
            ]
        )


    if st.button(
        "Update Report",
        use_container_width=True
    ):

        connection = sqlite3.connect(
            "reports.db"
        )

        connection.execute(
            """
            UPDATE reports
            SET status = ?
            WHERE report_id = ?
            """,
            (
                new_status,
                selected_report[0]
            )
        )

        connection.commit()
        connection.close()

        st.success(
            "Report status updated successfully."
        )

        st.rerun()

else:

    st.info(
        "No reports available for management."
    )