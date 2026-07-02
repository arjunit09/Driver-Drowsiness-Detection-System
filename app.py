from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_session import Session
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import io
import base64
from datetime import datetime
from final_drowsiness import start, cv2
from threading import Thread
import control,time

# =======================================
# Flask App Config
# =======================================

app = Flask(__name__)
app.secret_key = "super_secret_key"

app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Detection thread
detection_thread = None

# =======================================
# Fake User Data
# =======================================

USERS = {
    "admin": "password123"
}

# =======================================
# Utility: Read Alert Log
# =======================================

def read_alert_log():

    path = "alert_log.txt"

    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "type"])

    logs = []

    with open(path, "r") as f:
        for line in f:

            if "DROWSINESS" in line:
                alert_type = "Drowsiness"

            elif "YAWN" in line:
                alert_type = "Yawn"

            else:
                continue

            try:
                timestamp = line.strip().split("]")[0].replace("[", "")
            except:
                timestamp = ""

            logs.append({
                "time": timestamp,
                "type": alert_type
            })

    return pd.DataFrame(logs)


# =======================================
# Utility: Generate Chart
# =======================================

def generate_chart():

    df = read_alert_log()

    if df.empty:
        return None

    counts = df["type"].value_counts()

    plt.figure(figsize=(5,3))
    counts.plot(kind="bar")

    plt.title("Total Alerts")
    plt.xlabel("Type")
    plt.ylabel("Count")

    buf = io.BytesIO()

    plt.tight_layout()
    plt.savefig(buf, format="png")

    buf.seek(0)

    img = base64.b64encode(buf.getvalue()).decode("utf-8")

    plt.close()

    return img


# =======================================
# Routes
# =======================================

@app.route("/")
def home():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():

    username = request.form.get("username")
    password = request.form.get("password")

    if username in USERS and USERS[username] == password:

        session["user"] = username
        flash("Login successful!", "success")

        return redirect(url_for("dashboard"))

    else:

        flash("Invalid username or password.", "danger")
        return redirect(url_for("home"))


@app.route("/logout")
def logout():

    session.clear()

    flash("Logged out successfully.", "info")

    return redirect(url_for("home"))


@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect(url_for("home"))

    df = read_alert_log()

    total_drowsy = df[df["type"] == "Drowsiness"].shape[0]
    total_yawn = df[df["type"] == "Yawn"].shape[0]

    chart = generate_chart()

    return render_template(
        "dashboard.html",
        username=session["user"],
        total_drowsy=total_drowsy,
        total_yawn=total_yawn,
        chart=chart,
        logs=df.to_dict(orient="records")
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


# =======================================
# Live Dashboard Data (AJAX)
# =======================================

@app.route("/dashboard_data")
def dashboard_data():

    if "user" not in session:
        return jsonify({"error": "unauthorized"}), 401

    df = read_alert_log()

    total_drowsy = df[df["type"] == "Drowsiness"].shape[0]
    total_yawn = df[df["type"] == "Yawn"].shape[0]

    chart = generate_chart()

    latest_logs = df.tail(10).to_dict(orient="records")

    return jsonify({
        "total_drowsy": total_drowsy,
        "total_yawn": total_yawn,
        "chart": chart,
        "logs": latest_logs
    })


# =======================================
# Main Control Page
# =======================================

@app.route("/Home_page")
def index():

    if "user" not in session:

        flash("Please login first", "warning")
        return redirect(url_for("home"))

    return render_template("index.html")


# =======================================
# Start Detection
# =======================================

@app.route("/start_detection")
def start_detection():

    global detection_thread

    if not control.detection_running:

        control.detection_running = True

        detection_thread = Thread(target=start)
        detection_thread.daemon = True
        detection_thread.start()

    return redirect(url_for("index"))


# =======================================
# Stop Detection
# =======================================

@app.route("/stop_detection")
def stop_detection():

    control.detection_running = False
    time.sleep(0.2)

    return redirect(url_for("index"))


# =======================================
# Run Flask App
# =======================================

if __name__ == "__main__":

    if not os.path.exists("alert_log.txt"):
        open("alert_log.txt","w").close()

    app.run(debug=False, use_reloader=False)