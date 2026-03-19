#!/usr/bin/env python3
"""
Quick diagnostic: fetch camera URL, save raw bytes, try to decode with OpenCV.
Saves files: `test_raw.jpg` (raw bytes) and `test_decoded.jpg` (decoded image, if any).
"""
import os
import sys
import socket
import urllib.request
import urllib.error
import time

DEFAULT_URL = os.getenv('CAMERA_URL', 'http://192.168.43.60/cam-lo.jpg')
OUT_RAW = 'test_raw.jpg'
OUT_DECODE = 'test_decoded.jpg'

print('Diagnostic test_fetch_camera')
print('Using URL:', DEFAULT_URL)

# Try to fetch using urllib
try:
    req = urllib.request.Request(DEFAULT_URL, headers={"User-Agent": "curl/7.0"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        code = getattr(resp, 'getcode', lambda: None)()
        ctype = resp.getheader('Content-Type') if hasattr(resp, 'getheader') else None
        data = resp.read()
        print('HTTP code:', code)
        print('Content-Type:', ctype)
        print('Bytes downloaded:', len(data))
        with open(OUT_RAW, 'wb') as f:
            f.write(data)
        print('Saved raw bytes to', OUT_RAW)
except urllib.error.HTTPError as e:
    print('HTTPError:', e.code, e.reason)
    sys.exit(2)
except urllib.error.URLError as e:
    print('URLError:', e.reason)
    sys.exit(2)
except socket.timeout:
    print('Socket timeout')
    sys.exit(2)
except Exception as e:
    print('Fetch error:', e)
    sys.exit(2)

# Try to decode with OpenCV
try:
    import cv2
    import numpy as np
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print('cv2.imdecode returned None (data may not be a valid image)')
    else:
        print('Decoded image shape:', img.shape)
        ok = cv2.imwrite(OUT_DECODE, img)
        print('Wrote decoded image to', OUT_DECODE, 'ok=', ok)
except Exception as e:
    print('OpenCV decode error (opencv may be missing?):', e)
    print('You can still inspect', OUT_RAW, 'manually.')

print('\nDone. Inspect the saved files and the printed headers.')
