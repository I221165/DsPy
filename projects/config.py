"""
Shared setup used by every project in this folder.

Why a separate file? So we configure Groq + DSPy in ONE place. Each project
just does `from config import setup_groq` instead of repeating the setup.
This is also good practice for a real GitHub repo.
"""

import os
import dspy


def setup_groq(model: str = "groq/llama-3.3-70b-versatile"):
    """Connect DSPy to Groq and return the LM. Reads the key from an env var
    so your secret key is NEVER written inside the code."""

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "\n  GROQ_API_KEY is not set.\n"
            "  In PowerShell run:  $env:GROQ_API_KEY = \"gsk_your_key\"\n"
            "  (see the project README), then run again.\n"
        )

    lm = dspy.LM(model, api_key=api_key)
    dspy.configure(lm=lm)
    return lm
