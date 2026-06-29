"""
Shared setup used by every project in this folder.

Why a separate file? So we configure Groq + DSPy in ONE place. Each project
just does `from config import setup_groq` instead of repeating the setup.
This is also good practice for a real GitHub repo.
"""

import os
from pathlib import Path
import dspy
from dotenv import load_dotenv

# Load keys from the .env file in the project root (one level up from /projects).
# This runs as soon as ANY project does `from config import setup_groq`, so your
# keys are available everywhere without typing them in each PowerShell window.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


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
