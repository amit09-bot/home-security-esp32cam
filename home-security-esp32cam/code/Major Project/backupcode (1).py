# p1.py
import os
# Suppress TensorFlow oneDNN warnings (informational only)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import urllib.request
import urllib.error
import socket
import numpy as np
from PIL import Image
import io
import concurrent.futures
import time
import requests
import threading

CAMERA_URL = os.getenv('CAMERA_URL', 'http://192.168.43.60/cam-lo.jpg')
# Local fallback image
FALLBACK_IMAGE = os.path.join(os.path.dirname(__file__), 'test.jpg')
url = CAMERA_URL

# Telegram Bot configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '8086569433:AAGR324m_c3HHqyvUe-mWo34Fa4pVykAg0E')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '6470310842')

# Track last alert time
last_alert_time = 0
ALERT_COOLDOWN = 5  # seconds between alerts

# Store last successfully decoded frame
last_good_frame = None

############################################################
#                PERSON DETECTOR (HOG)                    
############################################################

# Initialize HOG detector at module level
try:
    _HOG_DETECTOR = cv2.HOGDescriptor()
    _HOG_DETECTOR.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
except Exception:
    _HOG_DETECTOR = None


def detect_people(frame):
    """Detect humans using OpenCV HOG detector (fast, no external models)."""
    if _HOG_DETECTOR is None:
        return [], [], []
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects, weights = _HOG_DETECTOR.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
        
        bboxes = []
        labels = []
        confs = []
        
        for (x, y, w, h), wt in zip(rects, weights):
            bboxes.append([int(x), int(y), int(x + w), int(y + h)])
            labels.append('person')
            # Map weight to confidence (0.0-1.0)
            try:
                c = float(wt)
                confs.append(min(max(1.0 / (1.0 + pow(2.718281828, -c)), 0.0), 1.0))
            except Exception:
                confs.append(0.5)
        
        return bboxes, labels, confs
    except Exception as e:
        print(f'[Detection] HOG error: {e}')
        return [], [], []


def draw_bbox(img, bboxes, labels, confidences):
    out = img.copy()
    for box, label, conf in zip(bboxes, labels, confidences):
        x1, y1, x2, y2 = box
        color = (0, 255, 0) if label == 'person' else (255, 0, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(out, text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out


############################################################
#                IMAGE FETCH (HTTP SNAPSHOT)               
############################################################

def fetch_image(url, timeout=5, retries=3, backoff=1, fallback_path=None):
    """Fetch image from URL with retries. Returns BGR image (numpy) or None."""
    global last_good_frame
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = resp.read()
            imgnp = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(imgnp, cv2.IMREAD_COLOR)
            if img is not None:
                last_good_frame = img.copy()
                return img
            else:
                print('Frame corrupted, skipping...')
                return None
        except Exception as e:
            print(f'Error fetching URL (attempt {attempt}/{retries}):', e)
            time.sleep(backoff * attempt)

    if fallback_path and os.path.isfile(fallback_path):
        img = cv2.imread(fallback_path)
        if img is not None:
            print('Using fallback local image:', fallback_path)
            return img

    print('No image available (network and fallback failed).')
    return None


############################################################
#                TELEGRAM ALERT FUNCTION                  
############################################################

def send_telegram_alert(message, image_frame=None):
    """Send a Telegram alert with optional image."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print('[Telegram] Bot token or chat ID missing.')
        return False

    try:
        # Send text message
        url_msg = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
        print(f'[Telegram Debug] Sending text: {message}')
        resp = requests.post(url_msg, json=payload, timeout=5)

        print(f'[Telegram Debug] Response: {resp.text}')

        # Send image
        if image_frame is not None:
            ok, img_encoded = cv2.imencode('.jpg', image_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ok:
                img_bytes = img_encoded.tobytes()
            else:
                print('[Telegram] OpenCV encoding failed, using PIL...')
                rgb = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                buf = io.BytesIO()
                pil_img.save(buf, format='JPEG', quality=95)
                img_bytes = buf.getvalue()
                buf.close()

            snapshot_path = os.path.join(os.path.dirname(__file__), 'last_alert.jpg')
            with open(snapshot_path, 'wb') as f:
                f.write(img_bytes)

            url_photo = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto'
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': 'Face Detection Snapshot'}
            files = {'photo': ('frame.jpg', img_bytes, 'image/jpeg')}
            resp_img = requests.post(url_photo, data=data, files=files, timeout=15)
            print(f'[Telegram Debug] Photo Response: {resp_img.text}')

        return True

    except Exception as e:
        print('[Telegram] Error:', e)
        return False


############################################################
#                WINDOWS: RAW DISPLAY                     
############################################################

def run1():
    cv2.namedWindow('live transmission', cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            im = fetch_image(url, timeout=5, retries=2, backoff=1, fallback_path=FALLBACK_IMAGE)
            if im is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, 'No image', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('live transmission', blank)
            else:
                cv2.imshow('live transmission', im)

            if (cv2.waitKey(100) & 0xFF) == ord('q'):
                break
    finally:
        cv2.destroyWindow('live transmission')


############################################################
#               WINDOWS: FACE DETECTION                   
############################################################

def run2():
    cv2.namedWindow('detection', cv2.WINDOW_AUTOSIZE)
    global last_alert_time, last_good_frame

    try:
        while True:
            im = fetch_image(url, fallback_path=FALLBACK_IMAGE)

            if im is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, 'No image', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('detection', blank)

                if last_good_frame is not None and (time.time() - last_alert_time > ALERT_COOLDOWN):
                    threading.Thread(
                        target=send_telegram_alert,
                        args=("⚠️ Frame corrupted — sending last good snapshot", last_good_frame.copy()),
                        daemon=True
                    ).start()
                    last_alert_time = time.time()

            else:
                bbox, label, conf = detect_people(im)
                im_with_bbox = draw_bbox(im, bbox, label, conf)
                cv2.imshow('detection', im_with_bbox)

                if 'person' in label:
                    current_time = time.time()
                    if current_time - last_alert_time > ALERT_COOLDOWN:
                        idx = label.index('person')
                        confidence = conf[idx]
                        msg = f'🚨 Person detected! Confidence: {confidence:.2%}'

                        threading.Thread(
                            target=send_telegram_alert,
                            args=(msg, im_with_bbox.copy()),
                            daemon=True
                        ).start()

                        last_alert_time = current_time

            if (cv2.waitKey(100) & 0xFF) == ord('q'):
                break

    finally:
        cv2.destroyWindow('detection')


############################################################
#                    PROGRAM START                        
############################################################

if __name__ == '__main__':
    print('Person detection started... press CTRL+C to stop')

    # Test Telegram
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        print("\nTesting Telegram connection…")
        send_telegram_alert("✅ Test alert from FACE detection app")
        time.sleep(2)

    print('[Startup] Starting detection loop.')

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(run1)
            f2 = executor.submit(run2)
            concurrent.futures.wait([f1, f2], return_when=concurrent.futures.FIRST_EXCEPTION)
    except KeyboardInterrupt:
        print('Program stopped.')
    finally:
        cv2.destroyAllWindows()
