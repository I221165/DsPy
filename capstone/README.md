# 🛒 VoltMart — AI Customer Support Agent (Capstone)

A production-style **RAG support agent** that answers customer questions using
**only** official store policies, escalates to a human when unsure, and traces
every answer so the support team can audit it.

This is the capstone that ties the whole stack together into one real problem.

## The problem

An online electronics store gets hundreds of repetitive questions about
shipping, returns, and warranties. Human agents are overwhelmed. We need an AI
assistant that is **accurate** (no made-up policies), **safe** (escalates when
unsure), **scalable** (thousands of policy docs), and **auditable** (every
answer logged).

## Architecture

```
   Customer question
          │
          ▼
   ┌──────────────┐   vector search   ┌──────────────────────┐
   │   PINECONE   │ ────────────────► │ top 5 candidate docs │
   │ (cloud DB)   │                   └──────────┬───────────┘
   └──────────────┘                              │ rerank by relevance
                                                 ▼
                                        ┌──────────────────┐
                                        │   COHERE rerank  │
                                        └────────┬─────────┘
                                                 │ best 3 policies
                                                 ▼
                                        ┌──────────────────┐
                                        │  DSPy + Groq     │  answer, or
                                        │  (grounded RAG)  │  escalate if unsure
                                        └────────┬─────────┘
                                                 ▼
                                          Customer reply
                                                 │
                              every step traced ▼
                                        ┌──────────────────┐
                                        │     LANGFUSE     │  (audit / debug)
                                        └──────────────────┘
```

## Why each tool earns its place

| Tool | Job in this app | Why it's needed |
|------|-----------------|------------------|
| **Pinecone** | cloud vector DB of all policies | a real store has thousands of docs — needs to scale |
| **Cohere rerank** | reorders results by relevance | wrong policy → wrong answer → angry customer |
| **DSPy + Groq** | grounded answering + escalation | answers from policies only, never guesses |
| **Langfuse** | traces every interaction | production support must be auditable & debuggable |

## Setup

```powershell
pip install -r ../requirements.txt
```

Add these keys to the `.env` in the project root (`f:\DSpy\.env`):

```
GROQ_API_KEY=gsk_...          # https://console.groq.com
PINECONE_API_KEY=...          # https://app.pinecone.io
COHERE_API_KEY=...            # https://dashboard.cohere.com
LANGFUSE_PUBLIC_KEY=pk-...    # optional — https://cloud.langfuse.com
LANGFUSE_SECRET_KEY=sk-...    # optional
```

(Langfuse is optional — the agent runs fine without it, just without tracing.)

## Run

```powershell
cd capstone
python support_agent.py
```

You'll see it answer real customer questions, cite which policy it used, and
**escalate** the one question the policies don't cover. If Langfuse keys are
set, every answer is recorded at cloud.langfuse.com → Tracing.

## What this demonstrates (for a portfolio / interview)

- Production RAG: **retrieve → rerank → generate**
- A **managed cloud vector database** (Pinecone)
- **Reranking** for retrieval quality (Cohere)
- **Grounded, safe** generation with an escalation fallback (DSPy)
- **Observability** for production monitoring (Langfuse)
