import subprocess
import requests
import os
import uuid
import time
import shutil

# =========================
# CONFIG
# =========================
PC_IP = "192.168.1.71"
PC_PORT = 9000

VIDEO_SECONDS = 5
WIDTH = 640
HEIGHT = 480
FPS = 10

# frame sampling rates (fps)
SAMPLING_RATES = [1, 2, 5]

OUT_DIR = "/home/pi/videos"
FRAME_DIR = "/home/pi/frames"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

DEVICE_ID = "raspberry_pi_cam_v3"

API_PROCESS_VIDEO = f"http://{PC_IP}:{PC_PORT}/api/process_video"
API_PROCESS_FRAMES = f"http://{PC_IP}:{PC_PORT}/api/process_frames"

# =========================
# RECORD VIDEO
# =========================
def record_video(path):
    cmd = [
        "rpicam-vid",
        "-t", str(VIDEO_SECONDS * 1000),
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--framerate", str(FPS),
        "--codec", "h264",
        "--profile", "baseline",
        "--nopreview",
        "-o", path
    ]
    subprocess.run(cmd, check=True)

# =========================
# EXTRACT FRAMES (FFMPEG)
# =========================
def extract_frames(video_path, out_dir, fps):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vf", f"fps={fps}",
        f"{out_dir}/frame_%03d.jpg"
    ]
    subprocess.run(cmd, check=True)

# =========================
# MAIN
# =========================
clip_id = str(uuid.uuid4())
video_path = os.path.join(OUT_DIR, f"clip_{clip_id}.mp4")

# ---------- RECORD ----------
print("[INFO] Recording video...")
ts_start_capture = int(time.time() * 1000)
record_video(video_path)
ts_end_capture = int(time.time() * 1000)

print(f"[INFO] Video saved: {video_path}")

# ---------- SEND VIDEO ----------
print("[INFO] Uploading FULL VIDEO...")

ts_start_upload = int(time.time() * 1000)

with open(video_path, "rb") as f:
    r = requests.post(
        API_PROCESS_VIDEO,
        files={"file": (os.path.basename(video_path), f, "video/mp4")},
        data={
            "clip_id": clip_id,
            "device_id": DEVICE_ID,
            "ts_start_capture": ts_start_capture,
            "ts_end_capture": ts_end_capture,
            "ts_start_upload": ts_start_upload,
            "mode": "video"
        },
        timeout=120
    )

ts_end_upload = int(time.time() * 1000)

print("[INFO] VIDEO RESPONSE:", r.status_code, r.text)

# ---------- FRAME SAMPLING ----------
for rate in SAMPLING_RATES:
    print(f"[INFO] Sampling frames @ {rate} fps")

    frames_path = os.path.join(FRAME_DIR, f"{clip_id}_{rate}fps")
    ts_start_sampling = int(time.time() * 1000)

    extract_frames(video_path, frames_path, rate)

    ts_end_sampling = int(time.time() * 1000)
    frames = sorted(os.listdir(frames_path))
    num_frames = len(frames)

    print(f"[INFO] {num_frames} frames extracted")

    files = []
    for fname in frames:
        files.append(
            ("frames", (fname, open(os.path.join(frames_path, fname), "rb"), "image/jpeg"))
        )

    ts_start_send = int(time.time() * 1000)

    r = requests.post(
        API_PROCESS_FRAMES,
        files=files,
        data={
            "clip_id": clip_id,
            "device_id": DEVICE_ID,
            "sampling_fps": rate,
            "num_frames": num_frames,
            "ts_start_sampling": ts_start_sampling,
            "ts_end_sampling": ts_end_sampling,
            "ts_start_send": ts_start_send,
            "mode": "frames"
        },
        timeout=120
    )

    ts_end_send = int(time.time() * 1000)

    print(
        f"[INFO] FRAMES RESPONSE @ {rate} fps:",
        r.status_code,
        r.text
    )

print("[INFO] ALL TESTS COMPLETED")
