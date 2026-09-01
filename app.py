import hashlib
import sqlite3
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO


# =========================
# Configuration
# =========================
MODEL_PATH = Path("best.pt")
DATABASE_PATH = Path("users.db")

CLASS_NAMES = [
    "bottle",
    "grass",
    "branch",
    "milk-box",
    "plastic-bag",
    "plastic-garbage",
    "ball",
    "leaf",
]


# =========================
# Load YOLO Model
# =========================
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "best.pt was not found. Put your trained YOLOv8 model file "
        "in the same folder as app.py and run the application again."
    )

model = YOLO(str(MODEL_PATH))


# =========================
# Database Setup
# =========================
conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
cursor = conn.cursor()

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL
    )
    """
)
conn.commit()


# =========================
# Authentication
# =========================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def signup(username: str, password: str) -> str:
    username = (username or "").strip()

    if not username or not password:
        return "Please fill all fields."

    cursor.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    if cursor.fetchone():
        return "Username already exists."

    cursor.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        (username, hash_password(password)),
    )
    conn.commit()

    return "Signup successful. You can now log in."


def login(username: str, password: str):
    username = (username or "").strip()

    if not username or not password:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            "Please enter username and password.",
        )

    cursor.execute(
        "SELECT 1 FROM users WHERE username = ? AND password = ?",
        (username, hash_password(password)),
    )

    if cursor.fetchone():
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            f"Welcome, {username}!",
        )

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        "Invalid username or password.",
    )


def logout():
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        "Logged out successfully.",
    )


# =========================
# YOLO Detection
# =========================
def predict_image(image):
    if image is None:
        return None, "Please upload an image."

    if isinstance(image, np.ndarray):
        input_image = Image.fromarray(image)
    else:
        input_image = image

    results = model.predict(source=input_image, conf=0.25, verbose=False)

    output_img = results[0].plot()
    labels = []

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if hasattr(model, "names"):
                class_name = model.names.get(cls_id, CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id))
            else:
                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)

            labels.append(f"{class_name} ({confidence:.2f})")
    else:
        labels.append("No objects detected.")

    return output_img, "\n".join(labels)


# =========================
# Gradio UI
# =========================
custom_css = """
body {
    background-color: #e6f2ff;
}

.gradio-container {
    background-color: #e6f2ff !important;
}

h1, h2, h3, p {
    color: #003366 !important;
}

button {
    border-radius: 10px !important;
    font-size: 16px !important;
}

textarea,
input {
    border-radius: 8px !important;
}
"""


with gr.Blocks(css=custom_css, title="River Trash Detection System") as demo:
    gr.Markdown("# 🌊 River Trash Detection System")
    gr.Markdown(
        "YOLOv8-based detection of floating trash in river images."
    )

    # Authentication
    with gr.Column(visible=True) as auth_section:
        with gr.Tab("🔑 Login"):
            login_user = gr.Textbox(label="Username")
            login_pass = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")

        with gr.Tab("📝 Sign Up"):
            signup_user = gr.Textbox(label="Create Username")
            signup_pass = gr.Textbox(label="Create Password", type="password")
            signup_btn = gr.Button("Sign Up")

        auth_status = gr.Textbox(label="Status", interactive=False)

    # Main application
    with gr.Column(visible=False) as main_app:
        gr.Markdown("## ♻️ Upload Image for Trash Detection")

        image_input = gr.Image(
            type="numpy",
            label="Upload River Image",
        )

        detect_btn = gr.Button("Detect Trash")
        output_image = gr.Image(label="Detection Output")
        output_text = gr.Textbox(
            label="Detected Objects",
            lines=8,
            interactive=False,
        )

        logout_btn = gr.Button("Logout")

    # Events
    signup_btn.click(
        signup,
        inputs=[signup_user, signup_pass],
        outputs=auth_status,
    )

    login_btn.click(
        login,
        inputs=[login_user, login_pass],
        outputs=[main_app, auth_section, auth_status],
    )

    detect_btn.click(
        predict_image,
        inputs=image_input,
        outputs=[output_image, output_text],
    )

    logout_btn.click(
        logout,
        outputs=[main_app, auth_section, auth_status],
    )


if __name__ == "__main__":
    demo.launch()
