import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

MODEL_PATH = "/Users/brianmason/Desktop/SafeSleep/runs/detect/train3/weights/best.pt"

model = YOLO(MODEL_PATH)
app = FastAPI()
cap = cv2.VideoCapture(0)
executor = ThreadPoolExecutor(max_workers=1)


def capture_and_detect() -> bytes | None:
    ret, frame = cap.read()
    if not ret:
        return None

    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]}: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    _, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes()


async def generate_frames():
    loop = asyncio.get_event_loop()
    while True:
        frame_bytes = await loop.run_in_executor(executor, capture_and_detect)
        if frame_bytes is None:
            break
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
