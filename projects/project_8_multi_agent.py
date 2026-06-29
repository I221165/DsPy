"""
============================================================================
 PROJECT 8 — MULTI-AGENT TEAM  (several specialists working together)
============================================================================
One agent is good. A TEAM is often better -- the same way a company has a
researcher, a writer, and an editor instead of one person doing everything.

Here three specialist agents collaborate to produce a short article:
    1. RESEARCHER -> gathers key facts about a topic
    2. WRITER     -> turns those facts into a paragraph
    3. CRITIC     -> reviews it and suggests one improvement
    -> then the WRITER revises using the critic's note.

Each "agent" is just a DSPy Module with its own job (Signature). The real
skill is ORCHESTRATING them -- wiring their outputs into each other's inputs.
============================================================================
"""

import dspy
from config import setup_groq

setup_groq()


# ── 1. DEFINE EACH SPECIALIST'S JOB (one Signature each) ───────────────────
class Research(dspy.Signature):
    """List the most important facts about the topic."""
    topic: str = dspy.InputField()
    facts: str = dspy.OutputField(desc="3-4 key facts as short bullet points")


class Write(dspy.Signature):
    """Write a clear, engaging short paragraph using the given facts."""
    topic: str = dspy.InputField()
    facts: str = dspy.InputField()
    paragraph: str = dspy.OutputField(desc="a single polished paragraph")


class Critique(dspy.Signature):
    """Give ONE specific, actionable suggestion to improve the paragraph."""
    paragraph: str = dspy.InputField()
    suggestion: str = dspy.OutputField(desc="one concrete improvement")


class Revise(dspy.Signature):
    """Rewrite the paragraph applying the suggestion."""
    paragraph: str = dspy.InputField()
    suggestion: str = dspy.InputField()
    improved: str = dspy.OutputField(desc="the improved paragraph")


# ── 2. THE ORCHESTRATOR: a Module that runs the whole team ─────────────────
class WritingTeam(dspy.Module):
    def __init__(self):
        super().__init__()
        # each specialist is its own module
        self.researcher = dspy.ChainOfThought(Research)
        self.writer = dspy.ChainOfThought(Write)
        self.critic = dspy.ChainOfThought(Critique)
        self.reviser = dspy.ChainOfThought(Revise)

    def forward(self, topic):
        facts = self.researcher(topic=topic).facts            # step 1
        draft = self.writer(topic=topic, facts=facts).paragraph  # step 2
        note = self.critic(paragraph=draft).suggestion        # step 3
        final = self.reviser(paragraph=draft, suggestion=note).improved  # step 4
        # return everything so we can SEE the teamwork
        return dspy.Prediction(facts=facts, draft=draft, suggestion=note, final=final)


# ── 3. RUN THE TEAM ────────────────────────────────────────────────────────
team = WritingTeam()
result = team(topic="Why honey bees are important")

print("STEP 1 - Researcher gathered facts:\n", result.facts, "\n")
print("STEP 2 - Writer's first draft:\n", result.draft, "\n")
print("STEP 3 - Critic's suggestion:\n", result.suggestion, "\n")
print("STEP 4 - Final, revised paragraph:\n", result.final)
