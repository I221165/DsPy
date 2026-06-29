# DSPy Starter (wired to Groq)

This folder has one annotated learning file, `01_dspy_starter.py`, that shows
**all four core DSPy concepts** in one tiny working example, using **Groq** as
the (free, fast) LLM.

Do these steps in order.

---

## Step 1 — Install DSPy

Open PowerShell in this folder and run:

```powershell
pip install dspy
```

That's the only library you need. (DSPy pulls in everything else.)

---

## Step 2 — Get a free Groq API key

1. Go to **https://console.groq.com**
2. Sign up (free).
3. Open **API Keys** in the left menu → **Create API Key**.
4. Copy the key (it starts with `gsk_...`). You only see it once, so copy it now.

Now tell your computer about the key. In **PowerShell**, run (paste your real key):

```powershell
$env:GROQ_API_KEY = "gsk_your_real_key_here"
```

⚠️ Note: that line only lasts for the *current* PowerShell window. If you close
it, set it again. (Later you can make it permanent — ask me when you want that.)

---

## Step 3 — Run it

In the **same** PowerShell window (so the key is still set):

```powershell
python 01_dspy_starter.py
```

You should see it connect to Groq, answer two questions, score itself, then
optimize and score again.

---

## What you're looking at

| Concept | Where in the file | Plain-English meaning |
|--------|-------------------|------------------------|
| **Signature** | `class BasicQA` | What goes IN, what comes OUT |
| **Module** | `dspy.Predict`, `dspy.ChainOfThought` | The program that calls the model |
| **Metric + Evaluate** | `answer_match`, `dspy.Evaluate` | How you measure if it works |
| **Optimizer** | `dspy.BootstrapFewShot` | DSPy auto-improves the prompt |

Learn these four and you understand most of DSPy.

---

## If something breaks

- **`GROQ_API_KEY is not set`** → redo Step 2 in the same window, then rerun.
- **`model not found` / decommissioned** → the Groq model name changed. Check
  https://console.groq.com for the current model and update the model string
  near the top of `01_dspy_starter.py`.
- **`No module named dspy`** → redo Step 1.
- Anything else → paste me the error and I'll sort it.
