"""
============================================================================
 PROJECT 1 — TEXT CLASSIFIER  (sentiment of customer reviews)
============================================================================
What you'll learn:
  - A Signature with a TYPED output (the model must pick from a fixed list)
  - dspy.Predict to run it
  - dspy.Evaluate to score it on labeled examples

This is the simplest useful DSPy program: text in -> one label out.
============================================================================
"""

from typing import Literal
import dspy
from config import setup_groq

setup_groq()  # connect to Groq


# ── 1. SIGNATURE ───────────────────────────────────────────────────────────
# The output is a Literal: the model is FORCED to choose one of these 3 words.
# That's how you make a reliable classifier instead of free-form text.
class ClassifyReview(dspy.Signature):
    """Classify the sentiment of a customer product review."""

    review: str = dspy.InputField(desc="the customer's review text")
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField(
        desc="overall sentiment of the review"
    )
 

# ── 2. MODULE ──────────────────────────────────────────────────────────────
classify = dspy.Predict(ClassifyReview)


# ── 3. TRY IT ──────────────────────────────────────────────────────────────
samples = [
    "This blender is amazing, best purchase I've made all year!",
    "It broke after two days. Total waste of money.",
    "It works. Nothing special, nothing wrong.",
]

print("=== Quick demo ===")
for text in samples:
    out = classify(review=text)
    print(f"  [{out.sentiment:8}]  {text}")
print()


# ── 4. EVALUATE — how accurate is it? ──────────────────────────────────────
# A small labeled test set (review -> correct sentiment).
testset = [
    dspy.Example(review="Absolutely love it, five stars!", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Terrible quality, returning it.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="It's okay, does the job.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="Exceeded my expectations, highly recommend.", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Stopped working within a week, very disappointed.", sentiment="negative").with_inputs("review"),
]

# Metric: correct if predicted label == true label.
def label_match(example, pred, trace=None):
    return example.sentiment == pred.sentiment

evaluator = dspy.Evaluate(devset=testset, metric=label_match, display_progress=True)

print("=== Accuracy on labeled test set ===")
score = evaluator(classify)
print("Score:", score, "(higher = more correct)")
