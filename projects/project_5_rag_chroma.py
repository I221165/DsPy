"""
============================================================================
 PROJECT 5 — REAL RAG BOT with a VECTOR DATABASE (Chroma)
============================================================================
This is Project 4 upgraded. Project 4 matched questions to facts by shared
WORDS (dumb). Here we use Chroma, which matches by MEANING (smart) — so a
question worded completely differently still finds the right fact.

What you'll learn:
  - What a vector database does (store text by meaning, search by meaning)
  - How Chroma plugs into the SAME DSPy module you already understand

NOTE: the FIRST time you run this, Chroma downloads a small embedding model
(~80MB), one time. After that it's instant. This needs no API key.
============================================================================
"""

import chromadb
import dspy
from config import setup_groq

setup_groq()  # Groq is still our LLM for the *answering* step


# ── 1. THE KNOWLEDGE BASE (same facts as Project 4) ────────────────────────
KNOWLEDGE = [
    "DSPy was created by researchers at Stanford NLP.",
    "In DSPy, a Signature defines the inputs and outputs of a task.",
    "A Module is the program that runs a task; examples are Predict and ChainOfThought.",
    "An Optimizer automatically improves prompts using example data.",
    "Groq is a provider that runs open models very fast and offers a free API tier.",
    "RAG stands for Retrieval-Augmented Generation: fetch facts, then answer.",
]


# ── 2. BUILD THE VECTOR DATABASE ───────────────────────────────────────────
# chromadb.Client() = an in-memory database (gone when the program ends).
# Chroma turns each fact into "meaning numbers" (embeddings) automatically.
print("Building the vector database (first run downloads a small model)...")
client = chromadb.Client()
collection = client.create_collection(name="dspy_facts")

# .add() embeds + stores every fact. Each needs a unique id.
collection.add(
    documents=KNOWLEDGE,
    ids=[str(i) for i in range(len(KNOWLEDGE))],
)
print("Stored", len(KNOWLEDGE), "facts.\n")


# ── 3. SEMANTIC RETRIEVER (the upgrade) ────────────────────────────────────
# Ask Chroma for the facts whose MEANING is closest to the question.
def retrieve(question: str, k: int = 2):
    results = collection.query(query_texts=[question], n_results=k)
    return results["documents"][0]   # list of the top-k matching facts


# ── 4. SAME DSPy PIECES AS PROJECT 4 ───────────────────────────────────────
class AnswerFromContext(dspy.Signature):
    """Answer the question using ONLY the given context. If unsure, say so."""

    context: str = dspy.InputField(desc="relevant facts to use")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="a short answer grounded in the context")


class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(AnswerFromContext)

    def forward(self, question):
        facts = retrieve(question)              # step 1: vector search
        context = "\n".join(facts)
        return self.generate(context=context, question=question)  # step 2: answer


# ── 5. SEE THE MEANING-SEARCH WIN ──────────────────────────────────────────
# Notice: these questions use DIFFERENT words than the facts.
#   "built"  vs the fact's "created"
#   "makes prompts better" vs the fact's "improves prompts"
# A keyword search would struggle. Vector search nails it.
bot = RAG()

questions = [
    "Who built DSPy?",                       # 'built' != 'created' in the fact
    "What makes the prompts better by itself?",  # different wording from 'optimizer'
    "What does RAG mean?",
]

print("=== RAG bot (semantic search) ===\n")
for q in questions:
    found = retrieve(q)                      # show what the DB pulled
    result = bot(question=q)
    print("Q:", q)
    print("   retrieved:", found)
    print("   answer   :", result.answer)
    print("-" * 70)
