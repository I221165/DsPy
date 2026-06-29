"""
============================================================================
 PROJECT 4 — MINI RAG Q&A BOT  (the most important real-world pattern)
============================================================================
RAG = Retrieval-Augmented Generation. The agent FETCHES relevant facts first,
then ANSWERS using them. This is how chatbots answer questions about YOUR
documents/notes instead of making things up.

What you'll learn:
  - A CUSTOM Module (your own class built from dspy.Module) — the real way
    DSPy programs are structured.
  - The retrieve -> reason -> answer pipeline.

No vector database needed yet: we use a tiny keyword retriever so it runs with
just DSPy + Groq. (Later we'll swap in a real vector DB like Chroma.)
============================================================================
"""

import dspy
from config import setup_groq

setup_groq()


# ── 1. A TINY "KNOWLEDGE BASE" ─────────────────────────────────────────────
# Pretend these are facts pulled from your company docs / notes.
KNOWLEDGE = [
    "DSPy was created by researchers at Stanford NLP.",
    "In DSPy, a Signature defines the inputs and outputs of a task.",
    "A Module is the program that runs a task; examples are Predict and ChainOfThought.",
    "An Optimizer (also called a teleprompter) automatically improves prompts using data.",
    "Groq is a provider that runs open models very fast and offers a free API tier.",
    "RAG stands for Retrieval-Augmented Generation: fetch facts, then answer.",
]


# ── 2. A SUPER-SIMPLE RETRIEVER ────────────────────────────────────────────
# Scores each fact by how many words it shares with the question, returns top-k.
# (A real system uses embeddings + a vector DB — same idea, smarter matching.)
def retrieve(question: str, k: int = 2):
    q_words = set(question.lower().split())
    scored = []
    for fact in KNOWLEDGE:
        overlap = len(q_words & set(fact.lower().split()))
        scored.append((overlap, fact))
    scored.sort(reverse=True)
    return [fact for score, fact in scored[:k]]


# ── 3. SIGNATURE: answer using provided context ────────────────────────────
class AnswerFromContext(dspy.Signature):
    """Answer the question using ONLY the given context. If unsure, say so."""

    context: str = dspy.InputField(desc="relevant facts to use")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="a short answer grounded in the context")


# ── 4. CUSTOM MODULE: wire retrieve + reason together ──────────────────────
# This is the big lesson: a real DSPy program is a class with a forward() method
# that chains steps. Here: retrieve facts, then answer with ChainOfThought.
class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(AnswerFromContext)

    def forward(self, question):
        facts = retrieve(question)              # step 1: get relevant facts
        context = "\n".join(facts)              # bundle them
        return self.generate(context=context, question=question)  # step 2: answer


# ── 5. USE IT ──────────────────────────────────────────────────────────────
bot = RAG()

questions = [
    "Who created DSPy?",
    "What does an optimizer do in DSPy?",
    "What does RAG stand for?",
]

print("=== RAG bot answers (grounded in the knowledge base) ===\n")
for q in questions:
    result = bot(question=q)
    print("Q:", q)
    print("A:", result.answer)
    print("-" * 60)
