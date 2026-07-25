from flask import Flask, request, jsonify, send_from_directory
import os
import sys
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

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/health')
def health():
    # The UI pings this on load so you can see the backend is up before demoing.
    return jsonify({"status": "ok"})


@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    position = request.form.get('position', 'PROP1')

    temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(temp_path)

    try:
        result = analyze_full_submission(temp_path, position=position)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)


if __name__ == '__main__':
    # Port 5001, not 5000: on macOS the AirPlay Receiver (Control Center) holds
    # port 5000, so http://localhost:5000 answers 403 before Flask ever sees the
    # request. 5001 avoids that entirely.
    port = int(os.environ.get("PORT", 5002))
    print(f"\n  Debate Judge Assistant  ->  http://localhost:{port}\n")
    # use_reloader=False: the reloader re-imports the pipeline (whisper +
    # sentence-transformers + mediapipe) on every file save and would kill an
    # analysis mid-request. Drop it if you want live reload while editing.
    app.run(debug=True, port=port, use_reloader=False)
