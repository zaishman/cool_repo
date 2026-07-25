import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcript_pipline"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_pipline"))

from transcribe import transcribe, split_into_sentences, merge_short_sentences, extract_audio_from_video
from structure_rules import check_structure_compliance, count_contentions
from scoring import detect_evidence_combined, count_filler_words, calculate_speaker_points
from arei_extraction import extract_arei, score_validity, summarize_argumentation, analyze_all_refutations
from mediapipe_analysis import analyze_video
from acoustic import analyse_acoustics, convert_to_wav

def analyse_speech(audio_path, position):
    audio_path = convert_to_wav(audio_path)
    raw_segments = transcribe(audio_path)
    full_text = " ".join(seg["text"] for seg in raw_segments)
    sentences = split_into_sentences(raw_segments)
    sentences = merge_short_sentences(sentences, min_words=17)
    acoustics = analyse_acoustics(audio_path, raw_segments= raw_segments)
    contention_count, boundaries = count_contentions(sentences)
    refutation_result = analyze_all_refutations(sentences)

    contentions_with_arei = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(sentences)
        contention_text = " ".join(sentences[start:end])

        if not contention_text.strip():   # NEW - skip empty slices entirely
            continue #checks IF the slice is empty so we dont keep on using empty api calls

        arei = extract_arei(contention_text)
        validity = score_validity(arei)
        evidence = detect_evidence_combined(contention_text)

        contentions_with_arei.append({
            "arei": arei,
            "validity": validity,
            "evidence": evidence
        })
        
    summary = summarize_argumentation(contentions_with_arei, clash_results= None)
    filler_result = count_filler_words(full_text)
    speech_duration = raw_segments[-1]["end"] if raw_segments else 0

    structure_result = check_structure_compliance(
        position=position,
        contention_count=contention_count,
        poi_count=0, poi_durations=[],     
        heckle_count=0, heckle_durations=[],
        speech_duration_sec=speech_duration
    )

    contentions_with_arei = [c for c in contentions_with_arei if c["arei"]["assertion"] is not None]
    contention_count = len(contentions_with_arei)
    
    validity_scores = [c["validity"]["evidence_supports_reasoning"] for c in contentions_with_arei] + \
        [c["validity"]["reasoning_supports_assertion"] for c in contentions_with_arei]

    evidence_flags = [c["evidence"]["has_evidence"] for c in contentions_with_arei]
    evidence_ratio = sum(evidence_flags) / len(evidence_flags) if evidence_flags else 0

    speaker_points = calculate_speaker_points(
        emotion_label=acoustics.get("emotion_label", "composed"),  # see note below
        filler_count=filler_result["total_fillers"],
        speech_word_count=len(full_text.split()),
        validity_scores=validity_scores,
        evidence_ratio=evidence_ratio,
        structure_complaint=structure_result["complaint"],
        clarity_score=acoustics["clarity"],
        projection_label=acoustics["projection"]["projection_label"],
        refutation_strength=[r["refutation_strength"] for r in refutation_result["refutations"]]
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

def analyze_full_submission(video_path, position):
    wav_path = extract_audio_from_video(video_path)  
    speech_result = analyse_speech(wav_path, position)
    video_result = analyze_video(video_path)

    return {
        **speech_result,
        "video_analysis": video_result
    }

