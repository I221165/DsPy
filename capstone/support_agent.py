"""
============================================================================
 CAPSTONE — VoltMart AI CUSTOMER SUPPORT AGENT
============================================================================
A production-style RAG support agent that answers customer questions ONLY
from official store policies, escalates to a human when unsure, and traces
every answer for the support team to audit.

THE STACK (each tool earns its place):
   Groq      -> fast, cheap LLM (the brain)
   Pinecone  -> cloud vector database storing all policies (scales to 1000s)
   Cohere    -> reranks retrieved policies so the BEST one is used (accuracy)
   DSPy      -> grounded answering + a built-in "escalate if unsure" safety net
   Langfuse  -> records every interaction so the team can audit/debug (optional)

PIPELINE:  question -> Pinecone search -> Cohere rerank -> DSPy answer

SETUP -- put these in the .env at the project root (f:\\DSpy\\.env):
   GROQ_API_KEY=...          https://console.groq.com
   PINECONE_API_KEY=...      https://app.pinecone.io
   COHERE_API_KEY=...        https://dashboard.cohere.com
   LANGFUSE_PUBLIC_KEY=...   (optional) https://cloud.langfuse.com
   LANGFUSE_SECRET_KEY=...   (optional)
============================================================================
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
import dspy

from knowledge_base import STORE_POLICIES

# load keys from the .env in the project root (one level up from /capstone)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ── 0. CHECK REQUIRED KEYS ─────────────────────────────────────────────────
for key in ("GROQ_API_KEY", "PINECONE_API_KEY", "COHERE_API_KEY"):
    if not os.environ.get(key):
        raise SystemExit(f"\n  {key} is not set. Add it to your .env file.\n")


# ── 1. OBSERVABILITY (Langfuse) -- optional, on if keys are present ─────────
LANGFUSE_ON = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))
if LANGFUSE_ON:
    os.environ.setdefault("LANGFUSE_HOST", "https://cloud.langfuse.com")
    from langfuse import get_client
    from openinference.instrumentation.dspy import DSPyInstrumentor
    langfuse = get_client()
    if langfuse.auth_check():
        DSPyInstrumentor().instrument()
        print("Langfuse tracing: ON\n")
    else:
        LANGFUSE_ON = False
        print("Langfuse keys invalid -- continuing without tracing\n")
else:
    print("Langfuse tracing: OFF (no keys) -- the agent still works\n")


# ── 2. THE LLM (Groq) ──────────────────────────────────────────────────────
dspy.configure(lm=dspy.LM("groq/llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"]))


# ── 3. PINECONE: load all policies into the cloud vector DB ────────────────
from pinecone import Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
INDEX_NAME = "voltmart-policies"

if not pc.has_index(INDEX_NAME):
    print("Creating Pinecone index (first run only, ~30s)...")
    pc.create_index_for_model(
        name=INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={"model": "llama-text-embed-v2", "field_map": {"text": "chunk_text"}},
    )
    time.sleep(10)

index = pc.Index(INDEX_NAME)
print("Loading policies into Pinecone...")
index.upsert_records(
    namespace="policies",
    records=[{"_id": p["id"], "chunk_text": p["text"]} for p in STORE_POLICIES],
)
time.sleep(8)  # let the uploads become searchable


# ── 4. COHERE: rerank retrieved policies by true relevance ─────────────────
import cohere
co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

def retrieve(question, k=5):
    """Pinecone vector search -> top-k candidate policies as {id, text}."""
    resp = index.search(namespace="policies", query={"inputs": {"text": question}, "top_k": k})
    return [{"id": h["fields"]["_id"] if "_id" in h["fields"] else h["_id"],
             "text": h["fields"]["chunk_text"]} for h in resp["result"]["hits"]]

def rerank(question, candidates, top_n=3):
    """Cohere rerank -> keep only the most relevant policies."""
    r = co.rerank(model="rerank-v3.5", query=question,
                  documents=[c["text"] for c in candidates], top_n=top_n)
    return [candidates[item.index] for item in r.results]


# ── 5. DSPy: grounded answering with a safety net ──────────────────────────
class SupportReply(dspy.Signature):
    """You are VoltMart's support assistant. Answer the customer using ONLY the
    provided policies. If the policies do not contain the answer, set
    answerable=False and do not guess."""
    policies: str = dspy.InputField(desc="the relevant store policies")
    question: str = dspy.InputField(desc="the customer's question")
    answerable: bool = dspy.OutputField(desc="true only if the policies actually answer it")
    answer: str = dspy.OutputField(desc="a friendly, accurate reply to the customer")


class SupportAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.reply = dspy.ChainOfThought(SupportReply)

    def forward(self, question):
        candidates = retrieve(question)              # Pinecone
        best = rerank(question, candidates)          # Cohere
        policies_text = "\n".join(f"[{c['id']}] {c['text']}" for c in best)
        out = self.reply(policies=policies_text, question=question)

        if not out.answerable:                        # the safety net
            answer = ("I'm not certain about that from our policies, so I'm "
                      "connecting you with a human agent who can help.")
        else:
            answer = out.answer
        return dspy.Prediction(answer=answer, sources=[c["id"] for c in best],
                               escalated=not out.answerable)


# ── 6. RUN IT on real customer questions ───────────────────────────────────
agent = SupportAgent()

questions = [
    "I bought headphones 3 weeks ago but never opened them. Can I get my money back?",
    "How long does it take to get a refund after I send something back?",
    "My laptop arrived with a cracked screen. What do I do?",
    "Do you accept cash when the package arrives?",
    "Can I change the color of a product I already ordered?",   # not in policies -> should escalate
]

for q in questions:
    print("=" * 72)
    print("CUSTOMER:", q)
    result = agent(question=q)
    print("AGENT   :", result.answer)
    print("sources :", result.sources, "| escalated:", result.escalated)
print("=" * 72)

if LANGFUSE_ON:
    langfuse.flush()
    print("\nEvery answer above was traced. Audit them at cloud.langfuse.com -> Tracing")
