from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import time
import uuid
import os
import json
import shutil
import cv2
import numpy as np
import onnxruntime as ort
import shutil
HAS_FFMPEG = shutil.which("ffmpeg") is not None
print("HAS_FFMPEG:", HAS_FFMPEG)


# ===============================
# CONFIG
# ===============================
ONNX_MODEL_PATH = "yolo_small_weights.onnx"
INPUT_SIZE = 640
VIOLENCE_CLASS_ID = 1

CONF_THRESHOLDS = [round(x / 10, 1) for x in range(1, 10)]  # 0.1 → 0.9
DEFAULT_THRESH = 0.5

VIDEO_IN = "videos/input"
VIDEO_OUT = "videos/output"
LOG_PATH = "logs/experiments.json"


# ===============================
# APP
# ===============================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def ui():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

# ===============================
# ONNX SESSION
# ===============================
session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=["DmlExecutionProvider", "CPUExecutionProvider"]
)


print("Available providers:", ort.get_available_providers())
print("Session providers:", session.get_providers())


input_name = session.get_inputs()[0].name
print("ONNX input shape:", session.get_inputs()[0].shape)

LAST_EVENT = None

# ===============================
# HELPERS
# ===============================
def evaluate_thresholds(outputs, thresholds):
    """
    outputs: Nx6 -> x1,y1,x2,y2,conf,cls
    """
    result = {t: False for t in thresholds}

    for *_, conf, cls in outputs:
        if int(cls) != VIOLENCE_CLASS_ID:
            continue
        for t in thresholds:
            if conf >= t:
                result[t] = True

    return result

def preprocess(frame: np.ndarray) -> np.ndarray:
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def append_log(entry: dict):
    data = []

    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r") as f:
                content = f.read().strip()
                if content:
                    data = json.loads(content)
        except json.JSONDecodeError:
            data = []

    data.append(entry)

    with open(LOG_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ===============================
# CORE INFERENCE
# ===============================
def run_video_inference(video_path: str, save_out: bool = True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    scale_x = w / INPUT_SIZE
    scale_y = h / INPUT_SIZE

    out = None
    out_path = None
    tmp_avi = None

    if save_out:
        out_path = f"{VIDEO_OUT}/{os.path.basename(video_path)}"
        tmp_avi = out_path.replace(".mp4", ".avi")

        out = cv2.VideoWriter(
            tmp_avi,
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (w, h)
        )

    violence_per_thresh = {t: False for t in CONF_THRESHOLDS}
    frames = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames += 1
        inp = preprocess(frame)
        outputs = session.run(None, {input_name: inp})[0][0]

        frame_results = evaluate_thresholds(outputs, CONF_THRESHOLDS)
        for t, v in frame_results.items():
            if v:
                violence_per_thresh[t] = True

        # Bounding box SOLO per threshold di default
        if save_out:
            for x1, y1, x2, y2, conf, cls in outputs:
                if int(cls) == VIOLENCE_CLASS_ID and conf >= DEFAULT_THRESH:
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if out:
            out.write(frame)

    cap.release()
    if out:
        out.release()

        if HAS_FFMPEG:
            cmd = (
                f"ffmpeg -y -loglevel error "
                f"-i {tmp_avi} "
                f"-c:v libx264 -pix_fmt yuv420p {out_path}"
            )
            ret = os.system(cmd)
            if ret == 0 and os.path.exists(out_path):
                os.remove(tmp_avi)
            else:
                out_path = tmp_avi
        else:
            out_path = tmp_avi

    elapsed_ms = int((time.time() - t0) * 1000)
    return violence_per_thresh, frames, elapsed_ms



# ===============================
# ENDPOINTS
# ===============================

# ---- FULL VIDEO ----
@app.post("/api/process_video")
async def process_video(
    file: UploadFile = File(...),
    clip_id: str = Form(...),
    device_id: str = Form(...),
    sampling_fps: int | None = Form(None),
    mode: str | None = Form(None),
    ts_start_capture: int = Form(...),
    ts_end_capture: int = Form(...),
    ts_start_upload: int = Form(...),
    ground_truth: str = Form(...),
):
    in_path = f"{VIDEO_IN}/{clip_id}.mp4"

    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    violence_map, frames, inf_ms = run_video_inference(in_path, save_out=True)

    main_prediction = (
        "Violence" if violence_map[DEFAULT_THRESH] else "NoViolence"
    )

    entry = {
        "clip_id": clip_id,
        "mode": mode,
        "device_id": device_id,
        "sampling_fps": 10 if sampling_fps is None else sampling_fps,
        "prediction": main_prediction,
        "predictions_by_threshold": {
            str(t): ("Violence" if violence_map[t] else "NoViolence")
            for t in CONF_THRESHOLDS
        },
        "num_frames": frames,
        "timings_ms": {
            "capture": ts_end_capture - ts_start_capture,
            "inference": inf_ms
        },
        "ground_truth": ground_truth,
        "is_correct": (main_prediction == ground_truth)
    }

    append_log(entry)

    global LAST_EVENT
    LAST_EVENT = {
        "clip_id": clip_id,
        "mode": mode or "video",
        "device_id": device_id,
        "prediction": main_prediction,
        "confirmed": False,
        "ts_utc_ms": int(time.time() * 1000),
        "timings_ms": entry["timings_ms"]
    }

    return JSONResponse(entry)


# ---- FRAME SAMPLING ----
@app.post("/api/process_frames")
async def process_frames(
    clip_id: str = Form(...),
    device_id: str = Form(...),
    sampling_fps: int = Form(...),
    mode: str = Form(...),
    num_frames: int = Form(...),
    ts_start_sampling: int = Form(...),
    ts_end_sampling: int = Form(...),
    ts_start_send: int = Form(...),
    frames: list[UploadFile] = File(...),
    ground_truth: str = Form(...)
):
    violence_per_thresh = {t: False for t in CONF_THRESHOLDS}
    t0 = time.time()

    for f in frames:
        img_bytes = await f.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        inp = preprocess(img)
        outputs = session.run(None, {input_name: inp})[0][0]

        frame_results = evaluate_thresholds(outputs, CONF_THRESHOLDS)
        for t, v in frame_results.items():
            if v:
                violence_per_thresh[t] = True

    inf_ms = int((time.time() - t0) * 1000)

    main_prediction = (
        "Violence" if violence_per_thresh[DEFAULT_THRESH] else "NoViolence"
    )

    entry = {
        "clip_id": clip_id,
        "mode": mode,
        "device_id": device_id,
        "sampling_fps": sampling_fps,
        "num_frames": num_frames,
        "prediction": main_prediction,
        "predictions_by_threshold": {
            str(t): ("Violence" if violence_per_thresh[t] else "NoViolence")
            for t in CONF_THRESHOLDS
        },
        "timings_ms": {
            "sampling": ts_end_sampling - ts_start_sampling,
            "inference": inf_ms
        },
        "ground_truth": ground_truth,
        "is_correct": (main_prediction == ground_truth)
    }

    append_log(entry)
    return JSONResponse(entry)


# ---- LEGACY (UI) ----
@app.post("/api/process")
async def process_legacy(file: UploadFile = File(...)):
    clip_id = str(uuid.uuid4())
    in_path = f"{VIDEO_IN}/{clip_id}.mp4"

    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    violence, frames, elapsed = run_video_inference(in_path, save_out=True)

    global LAST_EVENT
    LAST_EVENT = {
        "clip_id": clip_id,
        "prediction": "Violence" if violence else "NoViolence",
        "confirmed": False,
        "mode": "video",
        "device_id": "PC",
        "ts_utc_ms": int(time.time() * 1000),
        "timings_ms": {
            "total": elapsed,
            "frames": frames
        }
    }

    return JSONResponse(LAST_EVENT)


# ---- UI ----
@app.get("/api/state")
def state():
    return {
        "server_mode": "PC-YOLO-ONNX",
        "last_event": LAST_EVENT
    }

@app.get("/api/view/{clip_id}")
def view_video(clip_id: str):
    mp4 = f"{VIDEO_OUT}/{clip_id}.mp4"
    avi = f"{VIDEO_OUT}/{clip_id}.avi"

    if os.path.exists(mp4):
        return FileResponse(mp4, media_type="video/mp4")

    if os.path.exists(avi):
        return FileResponse(avi, media_type="video/x-msvideo")

    return JSONResponse({"error": "video not ready"}, status_code=404)


@app.post("/api/confirm")
async def confirm_event(payload: dict):
    global LAST_EVENT

    if not LAST_EVENT:
        return {"ok": False, "error": "no event"}

    label = payload.get("label")
    if label not in ["violent", "not_violent"]:
        return {"ok": False, "error": "invalid label"}

    LAST_EVENT["confirmed"] = True
    LAST_EVENT["human_label"] = label

    return {"ok": True}

@app.post("/api/update_upload")
async def update_upload(
    clip_id: str = Form(...),
    mode: str = Form(...),            # "video" o "frames"
    sampling_fps: int | None = Form(None),
    upload_client_ms: int = Form(...)
):
    if not os.path.exists(LOG_PATH):
        return {"ok": False, "error": "log not found"}

    with open(LOG_PATH, "r") as f:
        data = json.load(f)

    updated = False
    for entry in reversed(data):
        if (
            entry["clip_id"] == clip_id
            and entry["mode"] == mode
            and (sampling_fps is None or entry.get("sampling_fps") == sampling_fps)
        ):
            if "timings_ms" not in entry:
                entry["timings_ms"] = {}

            entry["timings_ms"]["upload_client"] = upload_client_ms
            updated = True
            break

    if updated:
        with open(LOG_PATH, "w") as f:
            json.dump(data, f, indent=2)

    return {"ok": updated}


