"""
============================================================================
 PROJECT 7 — RESEARCH AGENT  (agent that uses a REAL web tool)
============================================================================
Project 6's tools were local (math, text). Real agents reach out to the
INTERNET. Here the agent gets a Wikipedia tool and decides, on its own, when
to look something up before answering.

This is the pattern behind "AI that can research": give it a search tool,
let it pull live facts, then answer grounded in what it found.

Uses Wikipedia's free public API -- no API key needed.
============================================================================
"""

import requests
import dspy
from config import setup_groq

setup_groq()


# ── 1. A REAL WEB TOOL ─────────────────────────────────────────────────────
def search_wikipedia(topic: str) -> str:
    """Look up a topic on Wikipedia and return a short factual summary.
    Use this whenever you need real-world facts you are unsure about."""
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(" ", "_")
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "dspy-learning-bot"})
        if r.status_code == 200:
            return r.json().get("extract", "No summary found.")
        return f"No Wikipedia page found for '{topic}'."
    except Exception as e:
        return f"Lookup failed: {e}"


# ── 2. THE RESEARCH AGENT ──────────────────────────────────────────────────
# Same idea as project 6, but now its tool reaches the live internet.
researcher = dspy.ReAct(
    "question -> answer",
    tools=[search_wikipedia],
    max_iters=5,
)


# ── 3. ASK IT THINGS IT MUST LOOK UP ───────────────────────────────────────
questions = [
    "What is the Eiffel Tower and roughly how tall is it?",
    "Who was Alan Turing and what is he known for?",
    "What is the capital of Australia?",   # many people get this wrong; the agent checks
]

for q in questions:
    print("=" * 70)
    print("QUESTION:", q)
    result = researcher(question=q)
    print("ANSWER:", result.answer)
print("=" * 70)
print("\n(The agent decided WHEN to call Wikipedia by itself -- that's the agentic part.)")
