"""
============================================================================
 PROJECT 6 — YOUR FIRST REAL AGENT  (tool use + reasoning loop)
============================================================================
This is the leap from "pipeline" to "AGENT".

Pipelines (projects 1-5): text in -> text out, one shot.
Agents (this): the model REASONS, decides "I need a tool", USES it, sees the
result, and decides what to do next -- in a LOOP -- until it has the answer.

DSPy gives you this for free with dspy.ReAct (Reason + Act).
You just hand it some TOOLS (plain Python functions) and a goal.
============================================================================
"""

import dspy
from config import setup_groq

setup_groq()


# ── 1. TOOLS = plain Python functions ──────────────────────────────────────
# The agent reads each function's NAME, DOCSTRING, and arguments to decide
# when to use it. So write clear docstrings -- that's how the agent "sees" them.

def calculator(expression: str) -> float:
    """Evaluate a math expression, e.g. '12 * 7 + 3'. Use for any arithmetic."""
    return eval(expression)   # fine for a learning demo; never eval untrusted input in real apps


def word_count(text: str) -> int:
    """Count how many words are in a piece of text."""
    return len(text.split())


def to_uppercase(text: str) -> str:
    """Convert text to UPPERCASE."""
    return text.upper()


# ── 2. THE AGENT ───────────────────────────────────────────────────────────
# Signature "question -> answer" is the overall goal.
# tools=[...] is the toolbox the agent can choose from.
agent = dspy.ReAct(
    "question -> answer",
    tools=[calculator, word_count, to_uppercase],
    max_iters=6,   # safety cap: at most 6 think/act steps
)


# ── 3. WATCH IT THINK AND ACT ──────────────────────────────────────────────
tasks = [
    "What is 1234 times 5678, then add 90?",
    "How many words are in the sentence: 'DSPy makes building agents easy'?",
    "Shout the phrase 'hello world' in uppercase.",
]

for task in tasks:
    print("=" * 70)
    print("TASK:", task)
    result = agent(question=task)
    print("ANSWER:", result.answer)
print("=" * 70)

# Want to SEE the agent's step-by-step reasoning and which tools it called?
# Uncomment the next line:
# dspy.inspect_history(n=5)
