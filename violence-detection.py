import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time, shlex, subprocess, threading, socket
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2, numpy as np

# =======================
# >>> ADD: TELEGRAM CONFIG <<<
# =======================
import requests

BOT_TOKEN = "8533554971:AAHhVWBFRgo9f5k5imcjFojJKloA-TR2r4Q"
CHAT_ID   = "726537758"

PI_GPS_PORT = 9100   # GPS socket from Pi
# =======================


# ----------- CONFIG -------------
PI_IP = "192.168.137.225"
PI_PORT = 9000
LAPTOP_LISTEN_IP = "0.0.0.0"
UDP_PORT = 5000

RECV_WIDTH, RECV_HEIGHT, RECV_FPS = 640, 360, 25
PROCESS_FPS = 6
PROCESS_INTERVAL = 1.0 / PROCESS_FPS

SEQ_LEN = 30
H, W = 96, 96
THRESHOLD = 0.55
SAVE_CLIP_SECONDS = 4
SAVE_COOLDOWN = 6.0
RESUME_DELAY = 5.0

DISPLAY_WINDOW = True
OUT_DIR = Path("detections"); OUT_DIR.mkdir(exist_ok=True)

FEATURE_TFLITE = Path("D:/PycharmProjects/convertion/split_models/feature_extractor_fp16.tflite")
LSTM_H5        = Path("D:/PycharmProjects/convertion/split_models/lstm_head.h5")
# ---------------------------------


# =======================
# >>> ADD: GPS FROM PI <<<
# =======================
def get_gps_from_pi(timeout=3):
    try:
        s = socket.create_connection((PI_IP, PI_GPS_PORT), timeout=timeout)
        data = s.recv(64).decode().strip()
        s.close()
        lat, lon = data.split(",")
        return lat, lon
    except:
        return "Unavailable", "Unavailable"
# =======================


# =======================
# >>> ADD: TELEGRAM SEND <<<
# =======================
def send_telegram_alert(frame, prob):
    lat, lon = get_gps_from_pi()

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    caption = (
        f"🚨 Violence Detected\n\n"
        f"Probability: {prob*100:.1f}%\n"
        f"Time: {ts}\n"
        f"Latitude: {lat}\n"
        f"Longitude: {lon}"
    )

    img_path = "alert.jpg"
    cv2.imwrite(img_path, frame)

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(img_path, "rb") as img:
        requests.post(
            url,
            data={"chat_id": CHAT_ID, "caption": caption, "parse_mode": "Markdown"},
            files={"photo": img}
        )
# =======================


# ---------- FFmpeg receiver (threaded) ----------
class FFmpegReceiver:
    def __init__(self, listen_ip="0.0.0.0", port=5000, width=320, height=180, fps=25):
        self.width=width; self.height=height; self.frame_size=width*height*3
        self.proc=None; self._lock=threading.Lock(); self._latest=None; self._running=True
        listen_url=f"udp://{listen_ip}:{port}"
        cmd = ("ffmpeg -hide_banner -loglevel error "
               "-fflags nobuffer -flags low_delay -probesize 32 -analyzeduration 0 "
               f"-i {listen_url} -f rawvideo -pix_fmt bgr24 "
               f"-vf scale={width}:{height} -r {fps} -")
        args = shlex.split(cmd)
        self.proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self._running:
            raw = self.proc.stdout.read(self.frame_size)
            if not raw: continue
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height,self.width,3))
            with self._lock:
                self._latest = frame

    def get_latest(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def stop(self):
        self._running=False
        if self.proc: self.proc.kill()


# ---------- Model loaders ----------
def load_tflite_interpreter(path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    interp = Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp


print("[INFO] Loading feature extractor...")
feat_interp = load_tflite_interpreter(FEATURE_TFLITE)
feat_in = feat_interp.get_input_details()[0]
feat_out = feat_interp.get_output_details()[0]

print("[INFO] Loading LSTM (.h5)")
from tensorflow.keras.models import load_model
lstm_model = load_model(str(LSTM_H5))
print("[INFO] Models loaded.")


def extract_feature(frame_bgr):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W,H)).astype(np.float32)/255.0
    arr = np.expand_dims(img, axis=0)
    feat_interp.set_tensor(feat_in['index'], arr)
    feat_interp.invoke()
    return feat_interp.get_tensor(feat_out['index']).reshape(-1)


# ---------- Networking: STOP/RESUME ----------
def send_cmd(cmd):
    try:
        s = socket.create_connection((PI_IP, PI_PORT), timeout=1)
        s.send(cmd.encode())
        s.close()
        print(f"[NET] Sent {cmd}")
    except:
        pass


# ---------- Main detection ----------
def main():
    receiver = FFmpegReceiver(LAPTOP_LISTEN_IP, UDP_PORT, RECV_WIDTH, RECV_HEIGHT, RECV_FPS)

    frame_buffer = deque(maxlen=SAVE_CLIP_SECONDS * RECV_FPS)
    feat_buffer = deque(maxlen=SEQ_LEN)

    last_process = 0
    last_detect_time = 0
    stopped_state = False
    last_alert = 0

    print("[INFO] Detection started. Waiting for frames...")

    while True:
        frame = receiver.get_latest()
        if frame is None:
            time.sleep(0.01)
            continue

        frame_buffer.append(frame.copy())
        now = time.time()

        if now - last_process < PROCESS_INTERVAL:
            if DISPLAY_WINDOW:
                cv2.imshow("Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        last_process = now
        feat = extract_feature(frame)
        feat_buffer.append(feat)

        if len(feat_buffer) == SEQ_LEN:
            seq = np.expand_dims(np.stack(feat_buffer), axis=0)
            prob = float(lstm_model.predict(seq, verbose=0)[0][0])
            print(f"[INFO] Violence prob: {prob:.3f}")

            if prob > THRESHOLD:
                last_detect_time = now

                if not stopped_state:
                    send_cmd("STOP")
                    stopped_state = True

                # >>> ADD: TELEGRAM ALERT (COOLDOWN) <<<
                if now - last_alert > SAVE_COOLDOWN:
                    send_telegram_alert(frame, prob)
                    last_alert = now

            if stopped_state and now - last_detect_time > RESUME_DELAY:
                send_cmd("RESUME")
                stopped_state = False
                feat_buffer.clear()
                frame_buffer.clear()

        disp = frame.copy()
        if len(feat_buffer) == SEQ_LEN:
            cv2.putText(disp, f"Violence: {prob*100:.1f}%", (8,26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0,0,255) if prob>THRESHOLD else (0,255,0), 2)

        if DISPLAY_WINDOW:
            cv2.imshow("Violence Detection", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    receiver.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
