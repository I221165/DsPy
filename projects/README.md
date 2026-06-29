# DSPy Projects

Four small, runnable DSPy projects (wired to Groq), from simplest to most
real-world. Each file is heavily commented — read it top to bottom as you run it.

## Setup (once per terminal)

```powershell
# from the f:\DSpy folder
$env:GROQ_API_KEY = "gsk_your_real_key_here"
cd projects
```

## The projects

| File | Project | Teaches |
|------|---------|---------|
| [project_1_classifier.py](project_1_classifier.py) | Review sentiment classifier | Typed outputs, `Predict`, `Evaluate` |
| [project_2_summarizer.py](project_2_summarizer.py) | TL;DR + key points summarizer | Multi-output Signatures, `ChainOfThought`, `inspect_history` |
| [project_3_math_solver.py](project_3_math_solver.py) | Math word-problem solver | **Optimizer** improving the score automatically |
| [project_4_rag_qa.py](project_4_rag_qa.py) | Mini RAG Q&A bot (keyword) | **Custom Module**, retrieve → reason → answer |
| [project_5_rag_chroma.py](project_5_rag_chroma.py) | Real RAG bot (vector DB) | **Chroma** vector database, semantic search by *meaning* |
| [project_6_agent_tools.py](project_6_agent_tools.py) | First real **agent** | `dspy.ReAct` — reason → use a tool → repeat |
| [project_7_research_agent.py](project_7_research_agent.py) | Research agent | agent uses a **real web tool** (live Wikipedia) |
| [project_8_multi_agent.py](project_8_multi_agent.py) | Multi-agent team | researcher + writer + critic collaborating |

## Run them

```powershell
python project_1_classifier.py
python project_2_summarizer.py
python project_3_math_solver.py
python project_4_rag_qa.py
python project_5_rag_chroma.py   # first run downloads a small embedding model (~80MB), one time
python project_6_agent_tools.py
python project_7_research_agent.py
python project_8_multi_agent.py
```

## Pipelines vs Agents

- **Projects 1–5** are *pipelines*: text in → text out, one shot.
- **Projects 6–8** are *agents*: they **decide**, **use tools**, and **loop** —
  this is what "agentic AI" actually means.

Start with project 1 and go in order — each builds on the ideas before it.

## Suggested next steps (make them your own)

- **Project 1:** add your own categories (e.g. classify support tickets as
  `billing` / `technical` / `account`).
- **Project 2:** paste in a real article and tweak the output fields.
- **Project 3:** add harder problems and watch how optimizing helps.
- **Project 4:** replace the knowledge base with facts about YOU or a topic you
  like — that's your first personal chatbot.

`config.py` holds the shared Groq connection so every project imports it from
one place.
