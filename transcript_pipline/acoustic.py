import parselmouth
from parselmouth.praat import call
from pydub import AudioSegment

def convert_to_wav(input_path, output_path="converted.wav"):
    audio = AudioSegment.from_file(input_path)  # pydub auto-detects format from the file itself
    audio.export(output_path, format="wav")
    return output_path

def analyse_acoustics(audio_path, raw_segments=None):
    sound = parselmouth.Sound(audio_path) # loads audio into praat's internal representation; and reads files independently

    pitch = sound.to_pitch() #runs praats tracking algo for pitch, and returns a object that we can extract values from 
    pitch_values = pitch.selected_array['frequency'] #gives frequency value per analysed time frame
    pitch_values = pitch_values[pitch_values != 0] # filters out unvoiced frames 

    pitch_mean = pitch_values.mean() if len(pitch_values) > 0 else 0 # calculate the pitch mean from the extracted values 
    pitch_std = pitch_values.std() if len(pitch_values) > 0 else 0 #get the std
    pitch_range = (pitch_values.max() - pitch_values.min()) if len(pitch_values) > 0 else 0 #get the pitch range

    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500) # call actually runs praat commands, and finds individual glottal pulses (of vocal fold vibration cycles within a typical human pitch range)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) # gets jitter and uses standard default params
    shimmer = call([sound, point_process], "Get shimmer (local)", 0,0, 0.0001, 0.02, 1.3, 1.6) #same here as above but for shimmer

    intensity = sound.to_intensity() #gets intensity or the eggastertion
    energy_mean = call(intensity, "Get mean", 0, 0, "energy")

    clarity_score = calculate_clarity_score(raw_segments)
    projection_result = calculate_projection_score(energy_mean)

    return {
        "pitch_mean_hz": pitch_mean,
        "pitch_std_hz": pitch_std,
        "pitch_range_hz": pitch_range,
        "jitter": jitter,
        "shimmer": shimmer,
        "energy_mean_db": energy_mean,
        "clarity": clarity_score,
        "projection": projection_result,
    }

def calculate_clarity_score(raw_segments):
    if not raw_segments:
        return 0
    avg_confidence = sum(seg["confidence"] for seg in raw_segments) / len(raw_segments)
    clarity_score = max(0, min(100, (avg_confidence + 1.0) * 100))
    return round(clarity_score, 1)


def calculate_projection_score(energy_mean_db):
    if energy_mean_db > 65:
        label = "strong projection"
    elif energy_mean_db > 55:
        label = "adequate projection"
    else: 
        label = "weak projections / needs to project more"

    return {"energy_db": energy_mean_db, "projection_label": label}
