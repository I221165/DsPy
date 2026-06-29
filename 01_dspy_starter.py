"""
============================================================================
 DSPy STARTER  —  all 4 core concepts in ONE small file, wired to GROQ
============================================================================

Read this top-to-bottom like a study guide. Every section maps to one of the
four ideas that make up ~80% of DSPy:

    1. SIGNATURE  -> what goes IN, what comes OUT (the task definition)
    2. MODULE     -> the program that DOES the task (Predict / ChainOfThought)
    3. METRIC + EVALUATE -> how you MEASURE if it's working
    4. OPTIMIZER  -> DSPy auto-improves the prompt for you

Run order:  read README.md first (get a free Groq key), then run this file.
============================================================================
"""

import os
import dspy
from dotenv import load_dotenv

load_dotenv()  # reads keys from the .env file in this folder


# ===========================================================================
# 0.  CONNECT TO THE LLM  (this is the Groq part)
# ---------------------------------------------------------------------------
# DSPy talks to many providers through one object: dspy.LM(...).
# The model string format is  "groq/<model-name>".
# We read the secret key from an environment variable so it's NOT written
# inside this file (never paste API keys into code you might share).
# ===========================================================================

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise SystemExit(
        "\n  GROQ_API_KEY is not set.\n"
        "  Open README.md and follow 'Step 2' to get a free key and set it.\n"
    )

# Pick any current Groq model. llama-3.3-70b is a strong free default.
# (If you ever get a 'model not found' error, check https://console.groq.com
#  for the exact current model name and swap it in below.)
lm = dspy.LM("groq/llama-3.3-70b-versatile", api_key=api_key)

# Tell DSPy: "use this LM for everything from now on."
dspy.configure(lm=lm)

print("Connected to Groq.\n")


# ===========================================================================
# 1.  SIGNATURE  —  the task definition
# ---------------------------------------------------------------------------
# A Signature says WHAT goes in and WHAT comes out. You do NOT write a prompt;
# you describe the job, and DSPy builds the prompt for you.
#
# Think of it like a labeled form:  inputs on top, outputs on the bottom.
# ===========================================================================

class BasicQA(dspy.Signature):
    """Answer the question with a short, factual answer."""  # <- this docstring
    #                                                            becomes guidance
    #                                                            for the model.

    question = dspy.InputField(desc="a question to answer")
    answer = dspy.OutputField(desc="a short factual answer")


# ===========================================================================
# 2.  MODULE  —  the program that runs the task
# ---------------------------------------------------------------------------
# A Module takes a Signature and actually CALLS the model.
# Two you'll use constantly:
#
#   dspy.Predict(Sig)         -> ask and get the answer directly
#   dspy.ChainOfThought(Sig)  -> SAME task, but the model reasons step-by-step
#                                first (usually more accurate on hard stuff)
# ===========================================================================

qa = dspy.Predict(BasicQA)

result = qa(question="What is the capital of Japan?")
print("[Predict]  Q: capital of Japan?")
print("           A:", result.answer, "\n")
#            ^^^^^^^^^^  you access outputs by the field name from the Signature.


# Same task, but ask the model to think first:
cot = dspy.ChainOfThought(BasicQA)
result2 = cot(question="If a train travels 60 km in 2 hours, what is its speed?")
print("[ChainOfThought]  Q: train speed?")
print("           reasoning:", result2.reasoning)   # CoT adds a 'reasoning' field
print("           A:", result2.answer, "\n")


# ===========================================================================
# 3.  METRIC + EVALUATE  —  measure if it's actually working
# ---------------------------------------------------------------------------
# To improve a program you must first MEASURE it. You need:
#   (a) some examples with known correct answers  (a tiny dataset)
#   (b) a metric: a function that returns True/1 when the answer is "good"
# ===========================================================================

# (a) a tiny dataset. dspy.Example wraps one labeled item.
#     .with_inputs("question") tells DSPy which field is the INPUT.
trainset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What is 5 plus 7?", answer="12").with_inputs("question"),
    dspy.Example(question="What color is the sky on a clear day?", answer="Blue").with_inputs("question"),
    dspy.Example(question="What is the capital of Italy?", answer="Rome").with_inputs("question"),
]

# (b) a metric. 'example' = the gold item, 'pred' = what our program produced.
#     Here: pass if the correct answer appears inside the model's answer.
def answer_match(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

# dspy.Evaluate runs our program over the dataset and scores it with the metric.
evaluator = dspy.Evaluate(devset=trainset, metric=answer_match, display_progress=True)

print("Scoring the plain Predict program:")
score_before = evaluator(qa)
print("Score before optimizing:", score_before, "\n")


# ===========================================================================
# 4.  OPTIMIZER  —  let DSPy improve the program automatically
# ---------------------------------------------------------------------------
# This is DSPy's superpower. An optimizer looks at your dataset + metric and
# automatically rewrites/augments the prompt (e.g. by picking good examples to
# show the model) to push the score UP — without you hand-editing prompts.
#
# BootstrapFewShot is the simplest one to start with.
# ===========================================================================

optimizer = dspy.BootstrapFewShot(metric=answer_match)

# .compile() returns a NEW, improved version of your program.
print("Optimizing... (this makes a few model calls, give it a moment)\n")
optimized_qa = optimizer.compile(student=dspy.Predict(BasicQA), trainset=trainset)

print("Scoring the OPTIMIZED program:")
score_after = evaluator(optimized_qa)
print("Score after optimizing:", score_after, "\n")

print("Done. You just saw all 4 DSPy concepts:")
print("  1. Signature   2. Module   3. Metric+Evaluate   4. Optimizer")
