import anthropic
import json
from scoring import detect_refutations

client = anthropic.Anthropic()

AREI_PROMPT = """You are analyzing one contention from a Public Forum debate speech. Extract the following components. If a component is missing, or unclear, use null.

Contention text: {contention_text}

Respond ONLY with valid JSON in this exact format, no other text:
{{
    "assertion": "the core claim being made, ideally 7-9 words",
    "reasoning": "the logical reason given for the assertion",
    "evidence": "the specific evidence/data/source cited or null if none",
    "impact": "the stated real-world consequence or why it matters"
}}"""

def extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No valid JSON object found in text: {text}")
    return text[start : end + 1]

def extract_arei(contention_text: str) -> dict:
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": AREI_PROMPT.format(contention_text=contention_text)}]
    )
    response_text = message.content[0].text
    clean_json_str = extract_json_object(response_text)
    return json.loads(clean_json_str)


VALIDITY_PROMPT = """Given this contention's components, rate on a scale of 1-5 whether the Evidence actually supports the Reasoning, and whether the Reasoning actually supports the Assertion. Be critical- a 5 means airtight logic, a 1 means a significant logical gap.

    Assertion: {assertion}
    Reasoning: {reasoning}
    Evidence: {evidence}

    Respond ONLY with valid JSON:
    {{
        "evidence_supports_reasoning": <1-5>,
        "reasoning_supports_assertion": <1-5>,
        "brief_explaination": "a 3-4 sentence paragraph on the weakest link, and why you ruled this way"
    }}"""

def score_validity(arei):
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": VALIDITY_PROMPT.format(**arei)}]
    )
    response_text = message.content[0].text
    clean_json_str = extract_json_object(response_text)
    return json.loads(clean_json_str)

SUMMARY_PROMPT = """Based on this structuredd debate analysis, write a brief summary of strengths and weakness. Base your summary ONLY on the data provided below- do not introduce new judgements not supported by these numbers:

Contention analyzed:
{contentions_json}

Clash/rebuttal matching results:
{clash_json}

Respond ONLY with valid JSON:
{{
    "strengths": ["point 1", "point 2"],
    "weaknesses": ["point 1", "point 2"],
    "overall_argumentation_score": <1-10, derived from the validity and clash scores above>
}}
"""

def summarize_argumentation(contentions_with_arei, clash_results):
    prompt = SUMMARY_PROMPT.format(
        contentions_json = json.dumps(contentions_with_arei, indent=2),
        clash_json= json.dumps(clash_results, indent=2)
    )

    message= client.messages.create(
        model= "claude-sonnet-4-6",
        max_tokens= 600,
        messages= [{"role": "user", "content": prompt}]
    )
    return json.loads(extract_json_object(message.content[0].text))

REFUTATION_PROMPT = """This sentence from a debate speech signals a refutation of an opposing argument:

"{refutation_sentence}"

Here is surrounding context from the same speech:
"{context}"

Identify what point is being refuted, and rate on a 1-5 scale how well the refutation
actually defends against or disproves that point (not just acknowledges it exists).
Be critical - a 5 means the refutation directly dismantles the opposing point with clear
reasoning; a 1 means it barely engages with the point at all.

Respond ONLY with valid JSON:
{{
  "point_being_refuted": "brief description",
  "refutation_strength": <1-5>,
  "brief_explanation": "one sentence on why"
}}
"""

def analyze_refutation(refutation_sentence, context):
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": REFUTATION_PROMPT.format(
            refutation_sentence=refutation_sentence, context=context
        )}]
    )
    return json.loads(extract_json_object(message.content[0].text))

def analyze_all_refutations(sentences):
    refutations = detect_refutations(sentences)
    results = []
    for r in refutations:
        start = max(0, r["index"] - 2)
        end = min(len(sentences), r["index"] + 3)
        context = " ".join(sentences[start:end])

        analysis = analyze_refutation(r["text"], context)
        results.append(analysis)
    return {
        "refutation_count": len(results),
        "refutations": results
    }