import os
import shutil
import subprocess
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcript_pipline"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_pipline"))

# mediatest.analyze_video is the holistic (face + pose) analyzer and is the one
# the report UI expects — it returns posture_stability and detection_ratio,
# which mediapipe_analysis.analyze_video does not. Importing both here used to
# shadow this one with the face-only version.
from mediatest import analyze_video
from holistic_overlay import generate_annotated_video
from transcribe import transcribe, split_into_sentences, merge_short_sentences, extract_audio_from_video
from structure_rules import check_structure_compliance, count_contentions
from scoring import detect_evidence_combined, count_filler_words, calculate_speaker_points
from arei_extraction import extract_arei, score_validity, summarize_argumentation, analyze_all_refutations
from acoustic import analyse_acoustics, convert_to_wav

def analyse_speech(audio_path, position):
    audio_path = convert_to_wav(audio_path)
    raw_segments = transcribe(audio_path)
    full_text = " ".join(seg["text"] for seg in raw_segments)
    sentences = split_into_sentences(raw_segments)
    sentences = merge_short_sentences(sentences, min_words=17)
    acoustics = analyse_acoustics(audio_path, raw_segments=raw_segments)
    contention_count, boundaries = count_contentions(sentences)
    refutation_result = analyze_all_refutations(sentences)

    contentions_with_arei = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(sentences)
        contention_text = " ".join(sentences[start:end])

        if not contention_text.strip():   # skip empty slices entirely
            continue  # checks IF the slice is empty so we don't waste empty api calls

        arei = extract_arei(contention_text)
        validity = score_validity(arei)
        evidence = detect_evidence_combined(contention_text)

        contentions_with_arei.append({
            "arei": arei,
            "validity": validity,
            "evidence": evidence
        })

    summary = summarize_argumentation(contentions_with_arei, clash_results=None)
    filler_result = count_filler_words(full_text)
    speech_duration = raw_segments[-1]["end"] if raw_segments else 0

    structure_result = check_structure_compliance(
        position=position,
        contention_count=contention_count,
        poi_count=0, poi_durations=[],
        heckle_count=0, heckle_durations=[],
        speech_duration_sec=speech_duration
    )

    # keep only contentions that actually produced real content, and recompute
    # the count from that filtered list so "found X contentions" reflects real
    # arguments, not empty transitional slices
    contentions_with_arei = [c for c in contentions_with_arei if c["arei"]["assertion"] is not None]
    contention_count = len(contentions_with_arei)

    validity_scores = [c["validity"]["evidence_supports_reasoning"] for c in contentions_with_arei] + \
        [c["validity"]["reasoning_supports_assertion"] for c in contentions_with_arei]

    evidence_flags = [c["evidence"]["has_evidence"] for c in contentions_with_arei]
    evidence_ratio = sum(evidence_flags) / len(evidence_flags) if evidence_flags else 0

    speaker_points = calculate_speaker_points(
        emotion_label=acoustics.get("emotion_label", "composed"),
        filler_count=filler_result["total_fillers"],
        speech_word_count=len(full_text.split()),
        validity_scores=validity_scores,
        evidence_ratio=evidence_ratio,
        structure_complaint=structure_result["complaint"], # fixed: was structure_complaint
        clarity_score=acoustics["clarity"],
        projection_label=acoustics["projection"]["projection_label"],
        refutation_strength=[r["refutation_strength"] for r in refutation_result["refutations"]]  # fixed: was refutation_strength (singular)
    )

    return {
        "position": position,
        "transcript": full_text,
        "sentences": sentences,
        "acoustics": acoustics,
        "contentions": contentions_with_arei,
        "filler_words": filler_result,
        "structure": structure_result,
        "summary": summary,
        "speaker_points": speaker_points,
        "refutations": refutation_result
    }

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _publish_original(video_path):
    """
    Copy (or transcode) the submitted recording into static/ so the report page
    can play it back. Browser-recorded webm and phone .mov files don't play
    everywhere, so hand them to ffmpeg when it's available.
    """
    original_dir = os.path.join(STATIC_DIR, "original")
    os.makedirs(original_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    dest = os.path.join(original_dir, f"{stem}.mp4")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        result = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", video_path,
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-movflags", "+faststart", dest],
            capture_output=True,
        )
        if result.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0:
            return f"/static/original/{os.path.basename(dest)}"

    dest = os.path.join(original_dir, os.path.basename(video_path))
    shutil.copyfile(video_path, dest)
    return f"/static/original/{os.path.basename(dest)}"


def _stage(name):
    print(f"    [{name}]", flush=True)


def analyze_full_submission(video_path, position):
    _stage("extracting audio")
    wav_path = extract_audio_from_video(video_path)

    _stage("transcribing + scoring speech (slowest stage)")
    speech_result = analyse_speech(wav_path, position)

    _stage("analyzing video landmarks")
    video_result = analyze_video(video_path)

    # NOTE: annotated-overlay video generation has been removed from this
    # live path on purpose — it processes every frame with no skipping and
    # was adding 10+ minutes to each request, causing the browser's fetch to
    # time out ("Failed to fetch"). It's a nice-to-have visual, not something
    # the report/scoring needs. If you want it later, run
    # holistic_overlay.generate_annotated_video(video_path, out_path)
    # manually on a specific file, outside of the live request path.

    original_url = None
    try:
        original_url = _publish_original(video_path)
    except Exception as e:
        print(f"[wiring] original video publish failed: {e}")

    _stage("rendering landmark overlay")
    stem = os.path.splitext(os.path.basename(video_path))[0]
    annotated_filename = f"annotated_{stem}.mp4"
    annotated_dir = os.path.join(STATIC_DIR, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)
    annotated_path = os.path.join(annotated_dir, annotated_filename)

    annotated_url = None
    try:
        if generate_annotated_video(video_path, annotated_path):
            annotated_url = f"/static/annotated/{annotated_filename}"
    except Exception as e:
        print(f"[wiring] annotated video failed: {e}")

    return {
        **speech_result,
        "video_analysis": video_result,
        "original_video_url": original_url,
        "annotated_video_url": annotated_url,
    }