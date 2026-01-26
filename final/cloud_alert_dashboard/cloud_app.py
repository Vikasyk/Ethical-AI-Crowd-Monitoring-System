from flask import Flask, request, jsonify, render_template
import sqlite3
from datetime import datetime

app = Flask(__name__)
DB = "alerts.db"

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            system_id TEXT,
            route_no TEXT,
            origin TEXT,
            destination TEXT,
            density REAL,
            threshold REAL,
            alert_type TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route("/api/alert", methods=["POST"])
def receive_alert():
    d = request.json
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
        INSERT INTO alerts
        (system_id, route_no, origin, destination, density, threshold, alert_type, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        d["system_id"], d["route_no"], d["origin"], d["destination"],
        d["density"], d["threshold"], d["alert_type"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()
    return jsonify({"status":"stored"})

@app.route("/")
@app.route("/alerts")
def dashboard():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT * FROM alerts ORDER BY timestamp DESC")
    alerts = c.fetchall()
    conn.close()
    return render_template("alerts.html", alerts=alerts)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
