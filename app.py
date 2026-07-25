import os

import sys

# MediaPipe's C++ core floods stderr with noise that has nothing to do with the
# analysis: a per-frame landmark_projection warning, and a telemetry uploader
# that fails to reach Google's "clearcut" endpoint and logs an ERROR about it
# every 60 seconds. Both are harmless; together they bury real tracebacks.
#
# These two variables are set first because TensorFlow does honour
# TF_CPP_MIN_LOG_LEVEL at load time. MediaPipe does not honour GLOG_minloglevel
# (it logs through absl, which ignores it), so quiet_logs filters fd 2 directly.
# Both must run before anything imports mediapipe — i.e. before `import wiring`.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import quiet_logs
quiet_logs.install()

import logging
import time
import traceback

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

sys.path.append(BASE_DIR)
from wiring import analyze_full_submission

# static_url_path="" serves frontend/ straight off the site root, so the UI and
# the API live on one origin and the browser never makes a cross-origin request.
# CORS stays on as a fallback for opening frontend/index.html directly.
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)


class NumpySafeJSONProvider(DefaultJSONProvider):
    """
    numpy scalars (np.float64, np.int64, np.bool_) are not JSON-serializable,
    and several pipeline stages — sklearn's emotion classifier, the landmark
    statistics — hand them back without noticing. Left alone this raises inside
    jsonify, so Flask returns an HTML 500 and the UI reports the useless
    "server returned a non-JSON response". Coerce them here, once, at the
    boundary, instead of chasing casts through every pipeline module.
    """

    @staticmethod
    def default(obj):
        if isinstance(obj, np.generic):     # any numpy scalar
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return DefaultJSONProvider.default(obj)


app.json = NumpySafeJSONProvider(app)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# static_folder above points at frontend/, so Flask's built-in /static route is
# not in play — the rendered videos need their own route.
MEDIA_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(os.path.join(MEDIA_DIR, "annotated"), exist_ok=True)
os.makedirs(os.path.join(MEDIA_DIR, "original"), exist_ok=True)


@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/static/<path:filename>')
def media(filename):
    # conditional=True gives HTTP range support, which <video> needs to seek.
    return send_from_directory(MEDIA_DIR, filename, conditional=True)


@app.route('/health')
def health():
    # The UI pings this on load so you can see the backend is up before demoing.
    return jsonify({"status": "ok"})


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file was uploaded."}), 400

    file = request.files['file']
    position = request.form.get('position', 'PROP1')

    temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(temp_path)

    # Werkzeug only logs a request once it has returned, so a multi-minute
    # analysis looks like a hung server. Announce it up front.
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    started = time.time()
    print(f"\n>>> ANALYZE  {file.filename}  ({size_mb:.1f} MB, {position}) — this takes a few minutes\n",
          flush=True)

    try:
        result = analyze_full_submission(temp_path, position=position)
    except Exception as e:
        # Print the real traceback: str(e) alone is often a bare message with no
        # indication of which pipeline stage died.
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    try:
        payload = jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Result could not be serialized: {type(e).__name__}: {e}"}), 500

    print(f"<<< DONE in {time.time() - started:.0f}s\n", flush=True)
    return payload


class _QuietHealthChecks(logging.Filter):
    """The UI polls /health every 15s; those lines drown out everything else."""

    def filter(self, record):
        msg = record.getMessage()
        return '/health' not in msg and '/static/' not in msg


logging.getLogger('werkzeug').addFilter(_QuietHealthChecks())


if __name__ == '__main__':
    # Port 5001, not 5000: on macOS the AirPlay Receiver (Control Center) holds
    # port 5000, so http://localhost:5000 answers 403 before Flask ever sees the
    # request. 5001 avoids that entirely.
    port = int(os.environ.get("PORT", 5003))
    print(f"\n  Debate Judge Assistant  ->  http://localhost:{port}\n")
    # use_reloader=False: the reloader re-imports the pipeline (whisper +
    # sentence-transformers + mediapipe) on every file save and would kill an
    # analysis mid-request. Drop it if you want live reload while editing.
    app.run(debug=True, port=port, use_reloader=False)
