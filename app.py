from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
from flask_session import Session
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import io
import base64
import re
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from final_drowsiness import start, generate_frames, cv2
from threading import Thread
import control,time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(CURRENT_DIR, "alert_log.txt")
DB_PATH = os.path.join(CURRENT_DIR, "users.db")

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
# SQLite Database Management
# =======================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()

    # Seed default admin user if empty
    cursor.execute('SELECT COUNT(*) FROM users')
    if cursor.fetchone()[0] == 0:
        default_hash = generate_password_hash("password123")
        cursor.execute(
            'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
            ("admin", "admin@example.com", default_hash)
        )
        conn.commit()
    conn.close()


def get_user_by_username(username):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user


def create_user(username, email, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    password_hash = generate_password_hash(password)
    try:
        cursor.execute(
            'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
            (username, email, password_hash)
        )
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    finally:
        conn.close()
    return success

# =======================================
# Utility: Read Alert Log
# =======================================

def read_alert_log(all_events=False):

    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=["time", "type"])

    logs = []

    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue

                if "DROWSINESS DETECTED" in line_str:
                    alert_type = "Drowsiness"
                elif "YAWN DETECTED" in line_str:
                    alert_type = "Yawn"
                elif all_events and "SYSTEM STARTED" in line_str:
                    alert_type = "System Started"
                elif all_events and "SYSTEM STOPPED" in line_str:
                    alert_type = "System Stopped"
                else:
                    continue

                match = re.search(r'\[(.*?)\]', line_str)
                timestamp = match.group(1) if match else ""

                logs.append({
                    "time": timestamp,
                    "type": alert_type
                })
    except Exception as e:
        print(f"Log reading exception handled safely: {e}")

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
# Authentication Middleware (Enforce Login on All URLs)
# =======================================

EXEMPT_ENDPOINTS = {"home", "login", "register", "static"}

@app.before_request
def require_login():
    endpoint = request.endpoint

    if not endpoint or endpoint == "static" or endpoint.startswith("static."):
        return

    if endpoint in EXEMPT_ENDPOINTS:
        return

    if "user" not in session:
        if request.path.startswith("/dashboard_data") or request.path.startswith("/clear_logs"):
            return jsonify({"error": "unauthorized"}), 401

        flash("Please log in first to access the system.", "warning")
        return redirect(url_for("home"))


# =======================================
# Routes
# =======================================

@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    user = get_user_by_username(username)

    if user and check_password_hash(user["password"], password):

        session["user"] = user["username"]
        flash(f"Login successful! Welcome back, {user['username']}.", "success")

        return redirect(url_for("index"))

    else:

        flash("Invalid username or password.", "danger")
        return redirect(url_for("home"))


@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not password:
            flash("Username and password are required.", "warning")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match.", "warning")
            return render_template("register.html")

        if get_user_by_username(username):
            flash("Username already taken. Please choose another.", "danger")
            return render_template("register.html")

        if create_user(username, email, password):
            flash("Account registered successfully! Please log in with your credentials.", "success")
            return redirect(url_for("home"))
        else:
            flash("Registration failed. Please try again.", "danger")
            return render_template("register.html")

    return render_template("register.html")


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

    total_drowsy = df[df["type"] == "Drowsiness"].shape[0] if not df.empty else 0
    total_yawn = df[df["type"] == "Yawn"].shape[0] if not df.empty else 0
    total_incidents = total_drowsy + total_yawn
    safety_score = max(0, 100 - (total_drowsy * 8 + total_yawn * 4))

    latest_logs = df.tail(15).to_dict(orient="records") if not df.empty else []
    latest_logs.reverse()

    return render_template(
        "dashboard.html",
        username=session["user"],
        total_drowsy=total_drowsy,
        total_yawn=total_yawn,
        total_incidents=total_incidents,
        safety_score=safety_score,
        detection_running=control.detection_running,
        logs=latest_logs
    )


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        flash(f"Thank you, {name if name else 'Driver'}! Your message has been sent successfully to the team.", "success")
        return redirect(url_for("contact"))
    return render_template("contact.html")


# =======================================
# Live Dashboard Data (AJAX)
# =======================================

@app.route("/dashboard_data")
def dashboard_data():

    if "user" not in session:
        return jsonify({"error": "unauthorized"}), 401

    df = read_alert_log()

    df_stats = read_alert_log(all_events=False)
    df_all = read_alert_log(all_events=True)

    total_drowsy = df_stats[df_stats["type"] == "Drowsiness"].shape[0] if not df_stats.empty else 0
    total_yawn = df_stats[df_stats["type"] == "Yawn"].shape[0] if not df_stats.empty else 0
    total_incidents = total_drowsy + total_yawn
    safety_score = max(0, 100 - (total_drowsy * 8 + total_yawn * 4))

    latest_logs = df_all.tail(20).to_dict(orient="records") if not df_all.empty else []
    latest_logs.reverse()

    return jsonify({
        "total_drowsy": total_drowsy,
        "total_yawn": total_yawn,
        "total_incidents": total_incidents,
        "safety_score": safety_score,
        "detection_running": control.detection_running,
        "logs": latest_logs
    })


@app.route("/clear_logs", methods=["POST"])
def clear_logs():

    if "user" not in session:
        return jsonify({"error": "unauthorized"}), 401

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w") as f:
            f.write("")

    return jsonify({"success": True})


# =======================================
# Main Control Page
# =======================================

@app.route("/Home_page")
def index():

    if "user" not in session:

        flash("Please login first", "warning")
        return redirect(url_for("home"))

    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# =======================================
# Start Detection
# =======================================

@app.route("/start_detection")
def start_detection():

    global detection_thread

    if detection_thread is not None and detection_thread.is_alive():
        return redirect(url_for("index"))

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

    init_db()

    if not os.path.exists(LOG_PATH):
        open(LOG_PATH,"w").close()

    app.run(debug=False, use_reloader=False)