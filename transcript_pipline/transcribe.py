from faster_whisper import WhisperModel
import re
from pydub import AudioSegment

model = WhisperModel("small", device="cpu", compute_type="int8")
# we define our whisper model as small, which has good accuracy and still fast enough, "int8" quantizes the model into 8-bit integers instead of full percision floats to double the speed on cpu without accuraccy loss

def transcribe(audio_path):
    segments, info = model.transcribe(audio_path, vad_filter=True)
    #vad is voice activity detection, tells whisper to skip silent parts and not to transcribe them
    # returns the segments of the speach and loops through it once to get text,start and end into a plain list of dicts
    results = []
    for seg in segments:
        results.append({
            "text": seg.text.strip(),
            "start": seg.start,
            "end": seg.end,
            "confidence": seg.avg_logprob
        }) #here we append the segments into results, like all of the data
    return results 

def split_into_sentences(transcribe_segments):
    sentences = []
    for seg in transcribe_segments:
        parts = re.split(r'(?<=[.!?])\s+', seg["text"]) # splits text into sentences by looking for whitespace after the punctuation
        for part in parts:
            if part.strip():
                sentences.append(part.strip())
    return sentences

def merge_short_sentences(sentences, min_words=6):
    merged = []
    buffer = ""

    for sentence in sentences:
        buffer = (buffer + " " + sentence).strip() if buffer else sentence
        if len(buffer.split()) >= min_words:
            merged.append(buffer)
            buffer = ""

    if buffer:
        merged.append(buffer)

    return merged


def extract_audio_from_video(video_path, output_wav_path="extracted_audio.wav"):
    audio = AudioSegment.from_file(video_path)  # pydub auto-detects the container format
    audio.export(output_wav_path, format="wav")
    return output_wav_path
