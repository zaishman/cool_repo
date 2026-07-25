from clash_matching import embedder, util
import re

SPEECH_RULES = {
    "PROP1": {"contentions_allowed": (1, 2), "new_points_allowed": True, "pois_allowed": True, "heckles_allowed": True, "can_refute": False},
    "OPP1":  {"contentions_allowed": (1, 2), "new_points_allowed": True, "pois_allowed": True, "heckles_allowed": True, "can_refute": True},
    "PROP2": {"contentions_allowed": (0, 0), "new_points_allowed": True, "pois_allowed": True, "heckles_allowed": True, "can_refute": True},
    "OPP2":  {"contentions_allowed": (0, 0), "new_points_allowed": True, "pois_allowed": True, "heckles_allowed": True, "can_refute": True},
    "PROP3": {"contentions_allowed": (0, 0), "new_points_allowed": False, "pois_allowed": False, "heckles_allowed": False, "can_refute": True},
    "OPP3":  {"contentions_allowed": (0, 0), "new_points_allowed": False, "pois_allowed": False, "heckles_allowed": False, "can_refute": True},
}

MAX_HECKLES_PER_SPEECH = 3
MAX_POIS_PER_SPEECH = 3
MAX_POI_SECONDS = 15
MAX_HECKLE_SECONDS = 10
SPEECH_LENGTH_MINUTES = 5

#made a lookup table keyed by speech position
#contentions allowed is a min, max tupple because some segements allow 1-2 new contents, while the rest dont introduce new ones (refuting/summarizing)
# put this into a dict so if a rule ever changes, it's an easy edut

def detect_contention_boundaries(sentences, shift_threshold=0.4, window_size=3):
    embeddings = embedder.encode(sentences, convert_to_tensor=True) #again, embedds human diction into number values

    boundaries = [0]
    for i in range(window_size, len(sentences)): #for each sentence, we compare it to the previous sentence's embedding with cos similarity; if two sentences are sematically close, then similairt stays high, when a new contention is introduced, similarity drops and becomes a shift signal 
        recent_window = embeddings[i - window_size:i].mean(dim=0)
        similarity = util.cos_sim(embeddings[i], recent_window).item()
        if similarity < shift_threshold:
            boundaries.append(i) #boundaries documents every time it shifts basically

    return boundaries

def merge_small_segments(boundaries, min_sentences=4):
    merged = [boundaries[0]]
    for b in boundaries[1:]:
        if b - merged[-1] >= min_sentences:
            merged.append(b)
    return merged

def count_contentions(sentences):
    boundaries = detect_contention_boundaries(sentences)
    boundaries = merge_small_segments(boundaries)
    return len(boundaries), boundaries #returns the count of the boundries, or change of contention for the arei extraction


def check_structure_compliance(position, contention_count, poi_count, poi_durations, heckle_count, heckle_durations, speech_duration_sec):
    rules = SPEECH_RULES[position] #defines the rules based on the speech slot (PROP2, PROP1, etc. etc.)
    violations = []

    min_c, max_c = rules["contentions_allowed"] #unpacks the tupple into min, and max, which are the values of the tupple in contentions allowed
    if not (min_c <= contention_count <= max_c):  #checks if min is less than or equal to, the contention count, came for max
        violations.append(f"Expected {min_c}-{max_c}, found {contention_count} contentions") #adds to the violation dict of the we needed _ contentions, but we got _ instead

    if not rules["pois_allowed"] and poi_count > 0: # checks if pois are allowe compared to our poi count
        violations.append(f"POIs not allowed in {position}, but {poi_count} found") #violates it
    elif poi_count > MAX_POIS_PER_SPEECH: # checks if the poi count is more than max pois
        violations.append(f"Exceeded max POIs: {poi_count}/{MAX_POIS_PER_SPEECH}") #violates
    for i, dur in enumerate(poi_durations): #checks if the pois are more than allowed in a time frame
        if dur > MAX_POI_SECONDS:
            violations.append(f"POI #{i+1} exceeded {MAX_POI_SECONDS}s: {dur:.1f}s") #violates

    if not rules["heckles_allowed"] and heckle_count > 0:
        violations.append(f"Heckles not allowed in {position}, but {heckle_count} found")
    elif heckle_count > MAX_HECKLES_PER_SPEECH:
        violations.append(f"Exceeded max heckles: {heckle_count}/{MAX_HECKLES_PER_SPEECH}")
    for i, dur in enumerate(heckle_durations):
        if dur > MAX_HECKLE_SECONDS:
            violations.append(f"Heckle #{i+1} exceeded {MAX_HECKLE_SECONDS}s: {dur:.1f}s")

    #SAME POI violation structure for heckles

    expected_sec = SPEECH_LENGTH_MINUTES * 60 #length of the speech
    if abs(speech_duration_sec - expected_sec) > 30: # absolute values the speech duration we have - the speech duration set
        violations.append(f"Speech length {speech_duration_sec:.0f}s is off from expected {expected_sec}s") #if exceeded, violates

    return {
        "position": position,
        "complaint": len(violations) == 0,
        "violations": violations
    }