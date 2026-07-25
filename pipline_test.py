# test_full_pipeline.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcript_pipline"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_pipline"))

#from wiring import analyse_speech

from scoring import detect_evidence_combined
#prop1_result = analyse_speech("/Users/aditidivakar/projects/debate_judges_app/test_recording2.m4a", position="PROP1")

#print("=== PROP1 ANALYSIS ===")
#print(prop1_result)

sentences = [
    "According to a 2021 study published in the Journal of School Health, schools that removed sugary drinks saw a 15% drop in reported obesity rates.",
    "This is a strong argument for change.",
    "Data from the CDC shows that childhood obesity has tripled since the 1970s.",
    "Everyone knows sugary drinks are bad for you.",
    "Researchers at Yale found that sugar consumption directly correlates with reduced classroom attention spans.",
    "I think this policy just makes sense.",
    "A 2019 survey of school nutritionists found that 68% supported removing sugary beverages from campus."
]

for s in sentences:
    result = detect_evidence_combined(s)
    print(result["has_evidence"], "-", s)
