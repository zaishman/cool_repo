"""
Filters MediaPipe's C++ log noise out of the server's stderr.

MediaPipe logs from its native core, not from Python, so `logging`,
`absl.logging.set_verbosity` and GLOG_minloglevel all fail to silence it —
recent builds log through absl, which ignores the GLOG variables. The only
thing that reliably works is filtering file descriptor 2 itself.

So: replace fd 2 with a pipe, read it on a background thread, and forward every
line to the real stderr *except* the handful of known-noise patterns. Anything
unrecognised — including every Python traceback — passes through untouched.

Set DEBATE_VERBOSE_LOGS=1 to disable the filter entirely.
"""

import os
import re
import sys
import threading

# Matched against each stderr line. Everything here is MediaPipe/TF startup
# chatter or telemetry with no bearing on the analysis.
NOISE = re.compile(
    r"""(
        portable_clearcut_uploader     # telemetry retrying a failed upload, every 60s
      | Source\ Location\ Trace
      | wireless/android/play/playlog
      | landmark_projection_calculator
      | inference_feedback_manager
      | face_landmarker_graph
      | gl_context\.cc
      | init-domain\.cc
      | XNNPACK\ delegate
      | Feedback\ manager\ requires
      | Class\ AVF(Frame|Audio)Receiver   # cv2/av dylib collision warning
      | may\ cause\ spurious\ casting
    )""",
    re.VERBOSE,
)


def _pump(read_fd, out_fd):
    with os.fdopen(read_fd, "rb", buffering=0) as pipe:
        buffered = b""
        while True:
            chunk = pipe.read(4096)
            if not chunk:
                break
            buffered += chunk
            # Keep any trailing partial line for the next read.
            *lines, buffered = buffered.split(b"\n")
            for line in lines:
                if not NOISE.search(line.decode("utf-8", "replace")):
                    os.write(out_fd, line + b"\n")


def install():
    """Redirect fd 2 through the filter. Safe to call once, at startup."""
    if os.environ.get("DEBATE_VERBOSE_LOGS"):
        return

    sys.stderr.flush()
    real_stderr = os.dup(2)          # keep the original so the thread can write to it
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, 2)             # everything written to fd 2 now enters the pipe
    os.close(write_fd)

    thread = threading.Thread(target=_pump, args=(read_fd, real_stderr), daemon=True)
    thread.start()

    # Line-buffered, so a traceback shows up immediately rather than sitting in
    # a block buffer while you stare at an apparently-hung server.
    sys.stderr = os.fdopen(2, "w", buffering=1, errors="replace")
