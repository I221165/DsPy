"""
============================================================================
 PROJECT 9 — OBSERVABILITY with LANGFUSE  (see inside your agents)
============================================================================
So far your agents are a black box: you see the final answer, not HOW they
got there. Langfuse is a dashboard that records EVERY step -- prompts, tool
calls, timings, token cost -- so you can replay and debug any run.

How it works:
  1. We "instrument" DSPy  -> it now reports every step automatically.
  2. We run normal DSPy code (nothing about the agents changes).
  3. The full trace shows up on https://cloud.langfuse.com

SETUP (needs a free account -- see the README):
  Set these in PowerShell before running:
    $env:LANGFUSE_PUBLIC_KEY = "pk-..."
    $env:LANGFUSE_SECRET_KEY = "sk-..."
    $env:GROQ_API_KEY        = "gsk_..."
============================================================================
"""

import os
import dspy
from config import setup_groq


# ── 1. CHECK THE LANGFUSE KEYS ARE SET ─────────────────────────────────────
if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
    raise SystemExit(
        "\n  Langfuse keys not set. In PowerShell run:\n"
        '    $env:LANGFUSE_PUBLIC_KEY = "pk-..."\n'
        '    $env:LANGFUSE_SECRET_KEY = "sk-..."\n'
        "  (See the README to get them free.) Then run again.\n"
    )

# Region: EU is the default. If you signed up on the US server, set
#   $env:LANGFUSE_HOST = "https://us.cloud.langfuse.com"
os.environ.setdefault("LANGFUSE_HOST", "https://cloud.langfuse.com")


# ── 2. CONNECT TO LANGFUSE + TURN ON TRACING FOR DSPY ──────────────────────
from langfuse import get_client
langfuse = get_client()

if not langfuse.auth_check():
    raise SystemExit("\n  Langfuse rejected the keys. Double-check them and the region/host.\n")

# This one line makes DSPy report every step to Langfuse automatically.
from openinference.instrumentation.dspy import DSPyInstrumentor
DSPyInstrumentor().instrument()

print("Tracing is ON. Running some agents...\n")


# ── 3. RUN NORMAL DSPY CODE (the agents don't change at all) ───────────────
setup_groq()

# (a) a simple reasoning call
qa = dspy.ChainOfThought("question -> answer")
print("Q1:", qa(question="Why is the sky blue? Answer in one sentence.").answer)

# (b) a tool-using agent (same idea as project 6)
def calculator(expression: str) -> float:
    """Evaluate a math expression like '45 * 12'."""
    return eval(expression)

agent = dspy.ReAct("question -> answer", tools=[calculator], max_iters=5)
print("Q2:", agent(question="What is 45 times 12, plus 100?").answer)


# ── 4. SEND THE TRACES ─────────────────────────────────────────────────────
# flush() makes sure everything is uploaded before the program exits.
langfuse.flush()

print("\nDone! Open your dashboard to see the full step-by-step traces:")
print("  https://cloud.langfuse.com  ->  your project  ->  Tracing")
print("You'll see the reasoning, the calculator tool call, timings, and token cost.")
