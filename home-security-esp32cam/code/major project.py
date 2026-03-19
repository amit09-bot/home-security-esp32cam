# p1.py
import os
# Suppress TensorFlow oneDNN warnings (informational only)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import matplotlib.pyplot as plt
import socket
import numpy as np
try:
    from cvlib.object_detection import draw_bbox
except Exception:
    # Provide a no-op fallback so the program can run without cvlib
    def draw_bbox(img, bboxes, labels, confs):
        return img
import concurrent.futures
import time
import requests
import threading
import io
import traceback
from PIL import Image
import argparse
import sys
from urllib.parse import urlparse

CAMERA_URL = os.getenv('CAMERA_URL', 'http://10.49.111.60/cam-lo.jpg')

# Local fallback image
FALLBACK_IMAGE = os.path.join(os.path.dirname(__file__), 'test.jpg')

# Telegram Bot configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '8380359708:AAH9ZiOIVejXzYXZx2SMWHx3_coj_Dg-vqU')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '5245068996')

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


############################################################
#                IMAGE FETCH (HTTP SNAPSHOT)               
############################################################

def fetch_image(url, timeout=10, retries=5, backoff=1, fallback_path=None):
    """Fetch image from URL with retries. Returns BGR image (numpy) or None.

    Uses `requests` with optional basic auth from env vars `CAMERA_USER`/`CAMERA_PASS` and
    a configurable `FETCH_USER_AGENT` header. Supports local file paths as fallback.
    """
    global last_good_frame

    # If url is a local file path, try to read it directly
    try:
        if url and os.path.isfile(url):
            if cv2 is None:
                print('[fetch_image] OpenCV missing; cannot read local file')
                return None
            img = cv2.imread(url)
            if img is not None:
                last_good_frame = img.copy()
                return img
    except Exception:
        pass

    # Allow overriding timeout/retries via env for quick debugging
    try:
        timeout = int(os.getenv('FETCH_TIMEOUT', timeout))
    except Exception:
        pass
    try:
        retries = int(os.getenv('FETCH_RETRIES', retries))
    except Exception:
        pass
    dump_on_error = os.getenv('FETCH_DUMP_ON_ERROR', '1') in ('1', 'true', 'yes')

    # Prepare requests session
    session = requests.Session()
    headers = {
        'User-Agent': os.getenv('FETCH_USER_AGENT', 'Mozilla/5.0'),
        'Accept': os.getenv('FETCH_ACCEPT', 'image/webp,image/apng,image/*,*/*;q=0.8'),
        'Accept-Language': os.getenv('FETCH_ACCEPT_LANGUAGE', 'en-US,en;q=0.9'),
    }
    # Optional referer header
    referer = os.getenv('CAMERA_REFERER')
    if referer:
        headers['Referer'] = referer

    cam_user = os.getenv('CAMERA_USER')
    cam_pass = os.getenv('CAMERA_PASS')
    auth = None
    auth_type = os.getenv('CAMERA_AUTH_TYPE', 'none').lower()
    if auth_type in ('basic', 'digest') and cam_user and cam_pass:
        try:
            from requests.auth import HTTPBasicAuth, HTTPDigestAuth
            if auth_type == 'basic':
                auth = HTTPBasicAuth(cam_user, cam_pass)
            else:
                auth = HTTPDigestAuth(cam_user, cam_pass)
        except Exception:
            auth = (cam_user, cam_pass)

    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, headers=headers, timeout=timeout, auth=auth)
            status = getattr(resp, 'status_code', None)
            if status != 200:
                print(f'Error fetching URL (attempt {attempt}/{retries}): HTTP {status} - {resp.reason}')
                try:
                    # show limited response headers for debugging
                    hdrs = dict(resp.headers)
                    print('Response headers:', {k: hdrs.get(k) for k in ('Server', 'WWW-Authenticate', 'Content-Type', 'Content-Length') if k in hdrs})
                except Exception:
                    pass
                # Optionally dump small response body for debugging (login pages, redirects)
                if dump_on_error:
                    try:
                        ts = int(time.time())
                        base = os.path.join(os.path.dirname(__file__), f'last_fetch_resp_{ts}')
                        # Save a small binary sample
                        with open(base + '.bin', 'wb') as bf:
                            bf.write(resp.content[:4096])
                        # Also save text if available (limited)
                        try:
                            txt = resp.text
                            with open(base + '.html', 'w', encoding='utf-8', errors='replace') as tf:
                                tf.write(txt[:2000])
                        except Exception:
                            pass
                        print(f'[fetch_image] Saved response sample to {base}.*')
                    except Exception:
                        pass

                time.sleep(backoff * attempt)
                continue

            data = resp.content
            imgnp = np.frombuffer(data, dtype=np.uint8)
            if cv2 is None:
                print('[fetch_image] OpenCV missing; cannot decode image')
                return None
            img = cv2.imdecode(imgnp, cv2.IMREAD_COLOR)
            if img is not None:
                last_good_frame = img.copy()
                return img
            else:
                print('[fetch_image] Decoding failed; frame corrupted')
                return None

        except requests.RequestException as e:
            print(f'Error fetching URL (attempt {attempt}/{retries}):', e)
            # Dump exception details for debugging
            if dump_on_error:
                try:
                    ts = int(time.time())
                    errfile = os.path.join(os.path.dirname(__file__), f'last_fetch_err_{ts}.txt')
                    with open(errfile, 'w', encoding='utf-8') as ef:
                        ef.write('Exception:\n')
                        ef.write(repr(e) + '\n\n')
                        ef.write('Traceback:\n')
                        ef.write(traceback.format_exc())
                    print(f'[fetch_image] Wrote exception debug to {errfile}')
                except Exception:
                    pass
            time.sleep(backoff * attempt)

    # Fallback to local file if provided
    if fallback_path and os.path.isfile(fallback_path):
        if cv2 is None:
            print('[fetch_image] OpenCV missing; cannot read fallback image')
            return None
        img = cv2.imread(fallback_path)
        if img is not None:
            print('Using fallback local image:', fallback_path)
            last_good_frame = img.copy()
            return img

    print('No image available (network and fallback failed).')
    return None


def probe_camera(target, ports=(80, 8080, 81, 8000), paths=('/cam-lo.jpg', '/snapshot.jpg', '/image.jpg', '/cgi-bin/video.jpg', '/jpg', '/capture', '/')):
    """Try to discover a reachable HTTP snapshot URL for the given target (IP or URL).

    Returns the first working full URL or None.
    """
    session = requests.Session()
    headers = {'User-Agent': os.getenv('FETCH_USER_AGENT', 'Mozilla/5.0')}

    # If target already looks like a full URL, try it first
    parsed = urlparse(target)
    candidates = []
    if parsed.scheme in ('http', 'https'):
        candidates.append(target)
    else:
        host = target
        for p in ports:
            base = f'http://{host}:{p}'
            for path in paths:
                # avoid double slashes
                full = base + (path if path.startswith('/') else '/' + path)
                candidates.append(full)

    # Add common fallback when target had scheme but no path
    if parsed.scheme in ('http', 'https') and not parsed.path:
        host = parsed.netloc
        for p in ports:
            for path in paths:
                candidates.append(f'{parsed.scheme}://{host}:{p}{path}')

    tried = 0
    for url in candidates:
        tried += 1
        try:
            resp = session.get(url, headers=headers, timeout=3, stream=True, allow_redirects=True)
            status = getattr(resp, 'status_code', None)
            ctype = resp.headers.get('Content-Type', '') if resp is not None else ''
            if status == 200 and (ctype.startswith('image') or int(resp.headers.get('Content-Length', '0')) > 500):
                print(f'[probe_camera] Found working URL: {url} (Content-Type: {ctype})')
                return url
            else:
                print(f'[probe_camera] Tried {url}: HTTP {status} Content-Type={ctype}')
                # optionally save small HTML body for analysis
                try:
                    data = resp.content[:2048]
                    ts = int(time.time())
                    base = os.path.join(os.path.dirname(__file__), f'probe_resp_{ts}')
                    with open(base + '.bin', 'wb') as bf:
                        bf.write(data)
                except Exception:
                    pass
        except Exception as e:
            if tried % 10 == 0:
                print(f'[probe_camera] Tried {tried} candidates so far...')
            # quiet failures expected when probing many ports
    print('[probe_camera] No working URL found.')
    return None


############################################################
#                TELEGRAM ALERT FUNCTION                  
############################################################

def send_telegram_alert(message, image_frame=None, debug_info=None):
    """Send a Telegram alert with optional image.

    Saves a timestamped local snapshot and writes debug info to a .txt file.
    The `debug_info` string is appended to the photo caption when sending.
    """
    # Always print locally for debug
    ts = int(time.time())
    print(f'[Telegram Debug] {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))} - {message}')
    if debug_info:
        print('[Telegram Debug] Info:', debug_info)

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print('[Telegram] Bot token or chat ID missing. Skipping network send.')
        # Still save snapshot locally if image provided
        if image_frame is not None:
            try:
                snapshot_path = os.path.join(os.path.dirname(__file__), f'last_alert_{ts}.jpg')
                ok, img_encoded = cv2.imencode('.jpg', image_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                if ok:
                    with open(snapshot_path, 'wb') as f:
                        f.write(img_encoded.tobytes())
                print(f'[Telegram Debug] Saved local snapshot: {snapshot_path}')
                if debug_info:
                    with open(snapshot_path + '.txt', 'w', encoding='utf-8') as tf:
                        tf.write(debug_info)
            except Exception as e:
                print('[Telegram] Local save error:', e)
        return False

    try:
        # Send text message first
        url_msg = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message}
        resp = requests.post(url_msg, json=payload, timeout=5)
        print(f'[Telegram Debug] sendMessage response: {resp.status_code} {getattr(resp, "text", "")[:200]}')

        # Send image with caption including debug_info
        if image_frame is not None:
            ok, img_encoded = cv2.imencode('.jpg', image_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if ok:
                img_bytes = img_encoded.tobytes()
            else:
                # fallback to PIL encoding
                rgb = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                buf = io.BytesIO()
                pil_img.save(buf, format='JPEG', quality=95)
                img_bytes = buf.getvalue()
                buf.close()

            # Save a local timestamped snapshot and debug info
            try:
                snapshot_path = os.path.join(os.path.dirname(__file__), f'last_alert_{ts}.jpg')
                with open(snapshot_path, 'wb') as f:
                    f.write(img_bytes)
                if debug_info:
                    with open(snapshot_path + '.txt', 'w', encoding='utf-8') as tf:
                        tf.write(debug_info)
                print(f'[Telegram Debug] Saved local snapshot: {snapshot_path}')
            except Exception as e:
                print('[Telegram] Local save error:', e)

            url_photo = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto'
            caption = 'Face Detection Snapshot'
            if debug_info:
                # keep caption short; include first 200 chars of debug
                caption = caption + '\n' + (debug_info[:200])
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption}
            files = {'photo': ('frame.jpg', img_bytes, 'image/jpeg')}
            resp_img = requests.post(url_photo, data=data, files=files, timeout=15)
            print(f'[Telegram Debug] sendPhoto response: {resp_img.status_code} {getattr(resp_img, "text", "")[:200]}')

        return True

    except Exception as e:
        print('[Telegram] Error:', e)
        return False


############################################################
#                WINDOWS: RAW DISPLAY                     
############################################################

def run1(url):
    if os.getenv('USE_GUI', '1') in ('1', 'true', 'yes'):
        cv2.namedWindow('live transmission', cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            im = fetch_image(url, timeout=5, retries=2, backoff=1, fallback_path=FALLBACK_IMAGE)
            if im is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, 'No image', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if os.getenv('USE_GUI', '1') in ('1', 'true', 'yes'):
                    cv2.imshow('live transmission', blank)
            else:
                if os.getenv('USE_GUI', '1') in ('1', 'true', 'yes'):
                    cv2.imshow('live transmission', im)

            if (cv2.waitKey(100) & 0xFF) == ord('q'):
                break
    finally:
        if os.getenv('USE_GUI', '1') in ('1', 'true', 'yes'):
            try:
                cv2.destroyWindow('live transmission')
            except Exception:
                pass


############################################################
#               WINDOWS: FACE DETECTION                   
############################################################

def run2(url):
    if os.getenv('USE_GUI', '1') in ('1', 'true', 'yes'):
        cv2.namedWindow('detection', cv2.WINDOW_AUTOSIZE)
    global last_alert_time, last_good_frame

    try:
        while True:
            im = fetch_image(url, fallback_path=FALLBACK_IMAGE)

            if im is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, 'No image', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if os.getenv('USE_GUI', '1') in ('1', 'true', 'yes'):
                    cv2.imshow('detection', blank)

                if last_good_frame is not None and (time.time() - last_alert_time > ALERT_COOLDOWN):
                    debug = f'camera={url} time={time.strftime("%Y-%m-%d %H:%M:%S")} reason=frame_corrupted'
                    threading.Thread(
                        target=send_telegram_alert,
                        args=("⚠ Frame corrupted — sending last good snapshot", last_good_frame.copy(), debug),
                        daemon=True
                    ).start()
                    last_alert_time = time.time()

            else:
                bbox, label, conf = detect_people(im)
                im_with_bbox = draw_bbox(im, bbox, label, conf)
                if os.getenv('USE_GUI', '1') in ('1', 'true', 'yes'):
                    cv2.imshow('detection', im_with_bbox)

                if 'person' in label:
                    current_time = time.time()
                    if current_time - last_alert_time > ALERT_COOLDOWN:
                        idx = label.index('person')
                        confidence = conf[idx]
                        msg = f'🚨 Person detected! Confidence: {confidence:.2%}'

                        debug = f'camera={url} time={time.strftime("%Y-%m-%d %H:%M:%S")} label=person confidence={confidence:.4f}'
                        threading.Thread(
                            target=send_telegram_alert,
                            args=(msg, im_with_bbox.copy(), debug),
                            daemon=True
                        ).start()

                        last_alert_time = current_time

            if (cv2.waitKey(100) & 0xFF) == ord('q'):
                break

    finally:
        if os.getenv('USE_GUI', '1') in ('1', 'true', 'yes'):
            try:
                cv2.destroyWindow('detection')
            except Exception:
                pass


############################################################
#                    PROGRAM START                        
############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Person detection from HTTP camera snapshot')
    parser.add_argument('--headless', action='store_true', help='Run without GUI windows')
    parser.add_argument('--probe', action='store_true', help='Probe common ports/paths to find a working camera snapshot URL')
    parser.add_argument('--camera', type=str, help='Camera URL or IP (overrides CAMERA_URL env var)')
    args = parser.parse_args()

    print('Person detection started... press CTRL+C to stop')

    # Respect headless flag (also allow env override USE_GUI=0)
    if args.headless:
        os.environ['USE_GUI'] = '0'

    # Test Telegram (non-blocking)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        print('\nTesting Telegram connection…')
        send_telegram_alert('✅ Test alert from FACE detection app')
        time.sleep(2)

    print('[Startup] Starting detection loop.')

    url = args.camera or os.getenv('CAMERA_URL', 'http://192.168.77.60/cam-lo.jpg')

    # Optionally probe for working URL if requested
    if args.probe:
        print('[Startup] Probing for camera URL...')
        host_candidate = url
        # if a full URL was passed, try its netloc first
        parsed = urlparse(url)
        if parsed.scheme in ('http', 'https') and parsed.netloc:
            host_candidate = parsed.geturl()
        found = probe_camera(host_candidate)
        if found:
            print(f'[Startup] Probe replaced URL with: {found}')
            url = found
        else:
            print('[Startup] Probe failed — continuing with original URL')

    try:
        # If headless, only run detection loop; otherwise run both display+detection
        use_gui = os.getenv('USE_GUI', '1') in ('1', 'true', 'yes')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            if use_gui:
                futures.append(executor.submit(run1, url))
            futures.append(executor.submit(run2, url))
            concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
    except KeyboardInterrupt:
        print('Program stopped.')
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass