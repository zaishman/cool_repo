from clash_matching import embedder, util
import re

EVIDENCE_PATTERNS = [
    r'according to',
    r'a study (by|from|conducted|at|indicates)',
    r'data (from|shows)',
    r'research(ers)? (by|from|shows|indicates|at|found)',
    r'statistics (from|show|by|at|indicates)',
    r'\b(19|20)\d{2}\b', #any bare 4-digit year for if anytime someone sites a year
    r'reported by',
    r'survery (by|from|found)',
    r'%',
    r'evidence (states|says|indicates)'
] #general evidence signal markers

def detect_evidence(sentence):
    sentence_lower = sentence.lower()
    matches = [pattern for pattern in EVIDENCE_PATTERNS if re.search(pattern, sentence_lower)] # checks if the pattern happens anywhere in the sentences
    return {
        "has_evidence": len(matches) > 0,
        "matched_patterns": matches
    }

EVIDENCE_EXAMPLES = [
    "According to a 2021 study, the results showed significant improvement.",
    "Data from the CDC indicates a rise in cases.",
    "Researchers found that the effect was consistent across trials.",
    "A survey conducted last year revealed similar patterns.",
    "The numbers clearly support this conclusion based on the report."   
    "Researchers at the university found a strong correlation between the two factors.",
] #evidence examples

evidence_example_embeddings = embedder.encode(EVIDENCE_EXAMPLES, convert_to_tensor= True)
#again, embed the examples 
def detect_evidence_semantic(sentence, threshold= 0.45):
    sentence_embedding = embedder.encode(sentence, convert_to_tensor= True)
    similarities = util.cos_sim(sentence_embedding, evidence_example_embeddings)
    max_similarity = similarities.max().item()
    return {
        "has_evidence": max_similarity >= threshold,
        "confidence": max_similarity
    } #for every new sentence, we embed it, and compare against all the evidence examples at once to find any co-relaiton or ismilarities


def detect_evidence_combined(sentence):
    regex_result = detect_evidence(sentence)
    semantic_result = detect_evidence_semantic(sentence)
    return {
        "has_evidence": regex_result["has_evidence"] or semantic_result["has_evidence"],
        "matched_patterns": regex_result["matched_patterns"],
        "semantic_confidence": semantic_result["confidence"]
    } #uses both evidence trackers to make one big evidence signal-er


FILLER_WORDS = [
    'um',
    'uh',
    'like',
    'you know', 
    'basically',
    'literally', 
    'i mean',
    'so yeah'
] #defining filler words

def count_filler_words(full_transcript_text):
    text_lower = full_transcript_text.lower()
    counts = {}
    total = 0
    for filler in FILLER_WORDS:
        pattern = r'\b' + re.escape(filler) + r'\b' #makes sure the string is treated like literal text, not regex syntax (specialy characters)
    # the \b makes sure that words a literallyed trated
        matches = re.findall(pattern, text_lower) # returns match as a list to find the count for a specific filler word
        counts[filler] = len(matches) 
        total += len(matches) #returns total
    return {"total_fillers": total, "breakdown": counts}

def calculate_speaker_points(emotion_label, filler_count, speech_word_count, validity_scores, evidence_ratio, structure_complaint, clarity_score, projection_label, refutation_strength):
    base_score = 60 #baseline for passable speech

    emotion_bonus = {
        "composed": 3,
        "assertive": 2,
        "hesitant": -1,
        "off_topic": -2,
    } #gives scores based on the emotion

    base_score += emotion_bonus.get(emotion_label, 0)

    filler_rate = filler_count / max(speech_word_count, 1) #here we normalize, or make the rates porportional for the rate of filler words in a timeframe
    if filler_rate > 0.05:
        base_score -= 6
    elif filler_rate > 0.02:
        base_score -= 3

    if validity_scores:
        avg_validity = sum(validity_scores) / len(validity_scores)
        base_score += (avg_validity - 3) * 3 #scores it using our arei validity scoring, and centers the adjustments around 3 (no penalty)

    base_score += evidence_ratio * 5 #evidence flagged sentence/total sentences

    if not structure_complaint:
        base_score -= 5

    base_score += (clarity_score - 80) * 0.15
    projection_bonus = {"strong projection": 4, "adequate projection": 0, "weak projection / needs to project more": -5}
    base_score += projection_bonus.get(projection_label, 0)

    if refutation_strength:
        avg_refutation = sum(refutation_strength) / len(refutation_strength)
        base_score += (avg_refutation - 3) * 2

    return max(0, min(82, round(base_score))) #clamps the final result so it can never go below 0 or above 82, no matter how the adjustments stack up

REFUTATION_PATTERNS = [
    r'the opposition (may |might |will )?(argue|claim|say|contend)',
    r'however,',
    r'in response to',
    r'on the contrary',
    r'this fails to (account for|address|consider)',
    r'this (argument|claim|point) (is flawed|overlooks|ignores)',
]

REFUTATION_EXAMPLES = [
    "The opposition may argue that students should have the freedom to choose what they drink.",
    "However, this claim fails to account for the health risks involved.",
    "In response to their argument, the evidence actually shows the opposite.",
    "This point overlooks the fact that schools already regulate many choices.",
]

refutation_example_embeddings = embedder.encode(REFUTATION_EXAMPLES, convert_to_tensor=True)

def detect_refutation_semantic(sentence, threshold=0.4):
    sentence_embedding = embedder.encode(sentence, convert_to_tensor=True)
    similarities = util.cos_sim(sentence_embedding, refutation_example_embeddings)
    return similarities.max().item() >= threshold

def detect_refutations(sentences, gap_threshold = 2):
    raw_hits = []
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        regex_hit = any(re.search(pattern, sentence_lower) for pattern in REFUTATION_PATTERNS)
        semantic_hit = detect_refutation_semantic(sentence)
        if regex_hit or semantic_hit:
            raw_hits.append(i)

    if not raw_hits:
        return []

    merged = [[raw_hits[0]]]
    for idx in raw_hits[1:]:
        if idx - merged[-1][-1] <= gap_threshold:
            merged[-1].append(idx)
        else:
            merged.append([idx])

    return [{"index": group[0], "text": sentences[group[0]]} for group in merged]