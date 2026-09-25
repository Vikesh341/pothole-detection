import sqlite3

connection = sqlite3.connect("reports.db")

cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS reports (
    report_id TEXT PRIMARY KEY,
    date_time TEXT,
    image TEXT,
    evidence_image TEXT,
    detection TEXT,
    confidence REAL,
    area_percentage REAL,
    severity TEXT,
    location TEXT,
    status TEXT
)
""")

connection.commit()
connection.close()

print("Database created successfully!")