# Running the demo

```bash
python3 app.py
```

Then open **http://localhost:5001**

Flask serves `frontend/` off the site root, so the UI and the API share one
origin — nothing else to start, no separate web server.

## Before you demo — check these two things

1. **Top-right pill says "Backend connected."** If it says offline, the server
   isn't running.
2. **Anthropic credits.** The argument-extraction stage calls the Claude API. If
   the account is out of credits the pipeline gets ~86s in (transcription and
   acoustics finish fine) and then fails with:
   `Your credit balance is too low to access the Anthropic API`
   The UI shows that message verbatim on an "Analysis failed" screen with a retry.

## Flow

Landing → **Get started** → **New analysis** → pick a speech slot (PROP1…OPP3),
optionally label it → upload a file *or* record with camera & mic → progress
screen → report.

Reports are saved in browser `localStorage`, so once a run finishes you can
re-open it from the list instantly without re-running the pipeline. Handy for
demoing the report UI without waiting.

## Timing

A full pass on a 5-minute speech takes a few minutes: Whisper transcription,
Praat acoustics, sentence embeddings, one Claude call per contention plus one
per refutation plus a summary, and frame-by-frame MediaPipe on the video. The
progress screen's stage ticks are time-based estimates, not live backend events.

Large `.mov` files are the slow part — MediaPipe reads every frame. A shorter
clip demos much better.

## Notes

- Port is **5001**, not 5000: on macOS the AirPlay Receiver holds port 5000, so
  `localhost:5000` answers `403` before Flask sees the request. Override with
  `PORT=5002 python3 app.py`.
- The reloader is off (`use_reloader=False`) so saving a file mid-analysis can't
  kill the request or trigger a 30s+ model reload.
- One analysis at a time. The pipeline writes fixed temp filenames
  (`converted.wav`, `extracted_audio.wav`), so concurrent runs would overwrite
  each other; the UI blocks a second submit while one is in flight.
- Audio-only uploads work — you get the full transcript, delivery and
  argumentation breakdown, and the video section reports "No face detected."
