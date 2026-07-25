from clash_matching import match_rebuttals_to_claims
from arei_extraction import extract_arei, score_validity  # adjust import to wherever you saved these

claims = [
    "Universal basic income reduces poverty rates significantly.",
    "Government surveillance keeps citizens safer from terrorism."
]

rebuttals = [
    "UBI doesn't actually reduce poverty because it causes inflation that cancels out the benefit.",
    "Increased surveillance has historically led to discrimination against minority communities.",
    "The moon is made of rock and dust."  # deliberately unrelated, to test the threshold
]

print("=== CLASH MATCHING ===")
matches = match_rebuttals_to_claims(claims, rebuttals)
for m in matches:
    print(m)
    print()

print("=== AREI EXTRACTION ===")
sample_contention = """My first contention is that social media harms teen mental health.
Studies from the CDC show a 40% increase in teen depression rates since 2012, 
correlating directly with smartphone adoption. This means we're raising a generation
that struggles with anxiety and self-image at unprecedented rates."""

arei_result = extract_arei(sample_contention)
print(arei_result)

validity_result = score_validity(arei_result)
print(validity_result)