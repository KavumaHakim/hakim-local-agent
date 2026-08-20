"""Chooses which model should answer a given prompt.

Switching models on this hardware is expensive: unloading and reloading costs
roughly two minutes, and the new process starts with a cold prefix cache, so
the whole conversation is re-processed as well. The router is therefore
deliberately conservative:

* it scores the prompt with cheap heuristics before anything is loaded, so a
  correct guess costs nothing at all;
* it never routes *down*. Once a conversation has needed the strong model,
  going back would pay the switch cost twice to save RAM that is already
  spent;
* it is off unless the caller turns it on, and it always reports a reason so
  the interface can say why the model changed.

Getting it wrong is cheap in one direction only: starting on the small model
and escalating wastes one turn, while starting on the big model wastes minutes
on every trivial question. The thresholds lean small accordingly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Work that tends to need the stronger model. Matched as whole words.
HARD_SIGNALS = (
    "debug", "refactor", "architect", "architecture", "optimise", "optimize",
    "prove", "derive", "trace", "root cause", "diagnose", "design",
    "migrate", "benchmark", "profile", "review", "audit", "compare",
    "step by step", "step-by-step", "analyse", "analyze", "explain why",
    "why does", "why is", "plan", "strategy", "trade-off", "tradeoff",
    "implement", "rewrite", "algorithm", "complexity",
)

# Phrasing that is almost always a one-liner.
EASY_SIGNALS = (
    "hello", "hi", "hey", "thanks", "thank you", "what is", "what's",
    "who is", "when is", "where is", "define", "convert", "how many",
    "list the", "list files", "show me",
)

LONG_PROMPT = 250
VERY_LONG_PROMPT = 600
SHORT_PROMPT = 80

# Score at or above this routes to the strong model.
THRESHOLD = 3


@dataclass(frozen=True)
class RouteDecision:
    key: str
    reason: str
    score: int = 0

    def describe(self) -> str:
        return self.reason


class TaskRouter:
    """Picks a model key from the text of a prompt."""

    def __init__(
        self,
        fast_key: str,
        strong_key: str,
        *,
        enabled: bool = False,
        threshold: int = THRESHOLD,
    ) -> None:
        self.fast_key = fast_key
        self.strong_key = strong_key
        self.enabled = enabled
        self.threshold = threshold

    def choose(
        self,
        prompt: str,
        *,
        current_key: str | None = None,
        escalated: bool = False,
    ) -> RouteDecision:
        """Decide which model should handle `prompt`."""
        if not self.enabled:
            return RouteDecision(
                key=current_key or self.fast_key, reason="auto-routing off"
            )

        # Never route down: a conversation that has needed the strong model
        # keeps it rather than paying to switch back and forth.
        if escalated or current_key == self.strong_key:
            return RouteDecision(
                key=self.strong_key,
                reason="staying on the strong model for this conversation",
            )

        score, reasons = self.score(prompt)

        if score >= self.threshold:
            return RouteDecision(
                key=self.strong_key,
                score=score,
                reason="looks involved: " + ", ".join(reasons),
            )
        return RouteDecision(
            key=self.fast_key, score=score, reason="looks simple enough"
        )

    def score(self, prompt: str) -> tuple[int, list[str]]:
        """Rate how demanding a prompt looks. Higher means harder."""
        text = (prompt or "").strip()
        lowered = text.lower()
        score = 0
        reasons: list[str] = []

        if len(text) > VERY_LONG_PROMPT:
            # A prompt this size is a briefing, not a question.
            score += 3
            reasons.append("very long")
        elif len(text) > LONG_PROMPT:
            score += 1
            reasons.append("long")

        if "```" in text:
            # Pasted code almost always means read-and-reason work.
            score += 3
            reasons.append("contains code")

        if text.count("\n") > 5:
            score += 1
            reasons.append("many lines")

        hits = [word for word in HARD_SIGNALS if _mentions(lowered, word)]
        if hits:
            # Capped: three demanding words are not three times one.
            score += min(len(hits), 2) * 2
            reasons.append(f"mentions {hits[0]}")

        if lowered.count("?") > 1:
            score += 1
            reasons.append("several questions")

        if len(re.findall(r"[\w./\\-]+\.\w{1,4}\b", text)) >= 2:
            score += 1
            reasons.append("several files")

        # A short prompt opening with everyday phrasing is almost never work
        # for the big model, whatever else it happens to contain.
        if len(text) < SHORT_PROMPT and any(
            lowered.startswith(word) for word in EASY_SIGNALS
        ):
            score -= 2
            reasons.append("short and simple phrasing")

        return max(score, 0), reasons or ["nothing notable"]


def _mentions(haystack: str, phrase: str) -> bool:
    """Whole-word match, so 'plan' does not fire inside 'explanation'."""
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", haystack) is not None
