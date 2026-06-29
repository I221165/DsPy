"""
============================================================================
 PROJECT 10 — LIVE WEB SEARCH AGENT  (an agent that browses the internet)
============================================================================
Project 7's agent could only read Wikipedia. This one searches the LIVE
internet via DuckDuckGo -- so it can answer questions about recent events,
current prices, latest news, anything Wikipedia wouldn't have.

The agent DECIDES when to search, reads the results, and answers using them.
That "search -> read -> answer" loop is how real AI assistants stay current.

Uses DuckDuckGo -- free, no API key, no account.
============================================================================
"""

from ddgs import DDGS
import dspy
from config import setup_groq

setup_groq()


# ── 1. THE WEB SEARCH TOOL ─────────────────────────────────────────────────
# A plain function the agent can call. Returns the top results as text the
# model can read. Clear docstring = the agent knows when to use it.
def web_search(query: str) -> str:
    """Search the live internet and return the top results.
    Use this for recent events, current facts, news, or anything you're unsure of."""
    try:
        results = DDGS().text(query, max_results=4)
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No results found."

    # format the results into readable text for the model
    lines = []
    for r in results:
        lines.append(f"- {r['title']}: {r['body']}")
    return "\n".join(lines)


# ── 2. THE AGENT ───────────────────────────────────────────────────────────
agent = dspy.ReAct(
    "question -> answer",
    tools=[web_search],
    max_iters=6,   # it may search more than once before answering
)


# ── 3. ASK IT THINGS THAT NEED LIVE INFO ───────────────────────────────────
questions = [
    "What is DSPy and who maintains it?",
    "Give me a recent news headline about artificial intelligence.",
    "What programming language is the Django web framework written in?",
]

for q in questions:
    print("=" * 70)
    print("QUESTION:", q)
    result = agent(question=q)
    print("ANSWER:", result.answer)
print("=" * 70)
print("\n(The agent searched the web on its own, read results, then answered.)")

# To SEE the agent's searches and reasoning step by step, uncomment:
# dspy.inspect_history(n=8)
