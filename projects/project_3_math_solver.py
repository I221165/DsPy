"""
============================================================================
 PROJECT 3 — MATH WORD-PROBLEM SOLVER  (the DSPy "wow" project)
============================================================================
What you'll learn — DSPy's SUPERPOWER:
  We measure the program, then let an OPTIMIZER improve it automatically,
  and we WATCH the score go up — without hand-editing a single prompt.

  Flow:  build -> score -> optimize -> score again (and it improves).
============================================================================
"""

import re
import dspy
from config import setup_groq

setup_groq()


# ── 1. SIGNATURE ───────────────────────────────────────────────────────────
class SolveMath(dspy.Signature):
    """Solve the math word problem. Give the final numeric answer only."""

    problem: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="the final number, e.g. '42'")


# ── 2. METRIC ──────────────────────────────────────────────────────────────
# Pull the first number out of each answer and compare. This tolerates the
# model writing "The answer is 42." vs just "42".
def first_number(text: str):
    m = re.search(r"-?\d+\.?\d*", str(text))
    return m.group() if m else None

def numeric_match(example, pred, trace=None):
    return first_number(example.answer) == first_number(pred.answer)


# ── 3. DATA  (problem -> correct answer) ───────────────────────────────────
trainset = [
    dspy.Example(problem="Tom has 3 boxes with 4 apples each. How many apples total?", answer="12").with_inputs("problem"),
    dspy.Example(problem="A shirt costs $20 with a 25% discount. Final price?", answer="15").with_inputs("problem"),
    dspy.Example(problem="A car drives 150 km in 3 hours. Speed in km/h?", answer="50").with_inputs("problem"),
    dspy.Example(problem="Sara had 10 candies, ate 3, bought 5 more. How many now?", answer="12").with_inputs("problem"),
    dspy.Example(problem="A rectangle is 6 by 7. What is its area?", answer="42").with_inputs("problem"),
    dspy.Example(problem="There are 24 students split into 4 equal teams. Team size?", answer="6").with_inputs("problem"),
]
# Use a few for testing the score (kept separate from training is good practice;
# here we keep it simple and reuse the same small set).
testset = trainset

evaluator = dspy.Evaluate(devset=testset, metric=numeric_match, display_progress=True)


# ── 4. BEFORE: a plain ChainOfThought solver ───────────────────────────────
solver = dspy.ChainOfThought(SolveMath)

print("=== BEFORE optimizing ===")
score_before = evaluator(solver)
print("Score before:", score_before, "\n")


# ── 5. OPTIMIZE: let DSPy improve it automatically ─────────────────────────
# BootstrapFewShot finds good worked-examples and bakes them into the prompt.
print("=== Optimizing (a few Groq calls, give it a moment) ===\n")
optimizer = dspy.BootstrapFewShot(metric=numeric_match)
better_solver = optimizer.compile(student=dspy.ChainOfThought(SolveMath), trainset=trainset)


# ── 6. AFTER ───────────────────────────────────────────────────────────────
print("\n=== AFTER optimizing ===")
score_after = evaluator(better_solver)
print("Score after :", score_after)
print()
print(f"Result:  {score_before}  ->  {score_after}   (DSPy tuned the prompt for you)")
