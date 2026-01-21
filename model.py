'''
# model.py
import cv2
import time
import uuid
import numpy as np
import onnxruntime as ort

ONNX_MODEL_PATH = "yolo_small_weights.onnx"
INPUT_SIZE = 640
VIOLENCE_CLASS_ID = 1

session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=["CPUExecutionProvider"]  # CUDAExecutionProvider se disponibile
)
input_name = session.get_inputs()[0].name


def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def process_video(video_path, output_dir="videos/output", conf_thres=0.4):
    cap = cv2.VideoCapture(video_path)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    scale_x = w / INPUT_SIZE
    scale_y = h / INPUT_SIZE

    clip_id = str(uuid.uuid4())
    out_path = f"{output_dir}/{clip_id}.mp4"

    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    violence_found = False
    frames = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames += 1
        input_tensor = preprocess(frame)

        # output shape: (1, N, 6)
        outputs = session.run(None, {input_name: input_tensor})[0][0]

        for x1, y1, x2, y2, conf, cls in outputs:
            if int(cls) == VIOLENCE_CLASS_ID and conf >= conf_thres:
                violence_found = True

                # scale box to original frame
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    infer_time_ms = (time.time() - t0) * 1000

    return {
        "clip_id": clip_id,
        "prediction": "Violence" if violence_found else "NoViolence",
        "frames": frames,
        "time_ms": infer_time_ms,
        "video_path": out_path
    }
'''