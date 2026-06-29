"""
============================================================================
 PROJECT 2 — SMART SUMMARIZER  (TL;DR + key points)
============================================================================
What you'll learn:
  - A Signature with MULTIPLE outputs (the model fills in several fields)
  - dspy.ChainOfThought (model reasons before answering)
  - dspy.inspect_history() — SEE the exact prompt Groq received (debugging!)
============================================================================
"""

import dspy
from config import setup_groq

setup_groq()


# ── 1. SIGNATURE with several outputs ──────────────────────────────────────
# DSPy will make the model produce ALL of these fields in one go.
class Summarize(dspy.Signature):
    """Summarize the article for a busy reader."""

    article: str = dspy.InputField(desc="the full article text")

    tldr: str = dspy.OutputField(desc="one-sentence summary")
    key_points: str = dspy.OutputField(desc="3 short bullet points of the main ideas")
    reading_time_saved: str = dspy.OutputField(desc="rough estimate like '2 minutes'")


# ── 2. MODULE ──────────────────────────────────────────────────────────────
# ChainOfThought = the model thinks step-by-step first, then fills the fields.
summarizer = dspy.ChainOfThought(Summarize)


# ── 3. RUN IT ──────────────────────────────────────────────────────────────
article = """
DSPy is a framework for programming language models instead of prompting them
by hand. Instead of writing and tweaking long prompt strings, you describe your
task with a Signature (inputs and outputs), choose a Module to run it, and let
DSPy's optimizers automatically improve the prompts for you using example data.
This makes LLM programs more reliable, easier to maintain, and far less brittle
than copy-pasted prompt text. It works with many model providers and supports
building everything from simple classifiers to multi-step agents and RAG systems.
"""

out = summarizer(article=article)

print("=== Summary ===")
print("TL;DR        :", out.tldr)
print("Key points   :\n" + out.key_points)
print("Time saved   :", out.reading_time_saved)
print()

# ── 4. PEEK BEHIND THE CURTAIN ─────────────────────────────────────────────
# This prints the ACTUAL prompt DSPy built and sent to Groq, plus the reply.
# Get in the habit of using this whenever output surprises you.
print("=== What DSPy actually sent to Groq (last call) ===")
dspy.inspect_history(n=1)
