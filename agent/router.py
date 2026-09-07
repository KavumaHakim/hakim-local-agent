"""Chooses which model should answer a given prompt.

Switching models on this hardware is expensive: unloading and reloading costs
roughly two minutes, and the new process starts with a cold prefix cache, so
the whole conversation is re-processed as well. The router is therefore
deliberately conservative:

* it scores the prompt with cheap heuristics before anything is loaded, so a
  correct guess costs nothing at all;
* it never routes *down*. Once a conversation has needed a bigger model,
  going back would pay the switch cost twice to save RAM that is already
  spent;
* it is off unless the caller turns it on, and it always reports a reason so
  the interface can say why the model changed.

Getting it wrong is cheap in one direction only: starting on the small model
and escalating wastes one turn, while starting on the big model wastes minutes
on every trivial question. The thresholds lean small accordingly.

**An ordered chain, cheapest first.** It was a fast/strong pair, which is the
two-model case of the same idea; somebody with three models had no way to say
"try the 2B, then the 8B, then the hosted one". The chain is that list, and
the score picks a position in it: one `THRESHOLD` of score per step, capped at
the end. With two entries that is exactly the old rule - under the threshold
the first, at or over it the second - which is why the two-model tests did not
change.

A chain of one is a valid, and useful, way to say "never switch".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

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
    """Picks a model from an ordered chain, using the text of a prompt."""

    def __init__(
        self,
        chain: Sequence[str],
        *,
        enabled: bool = False,
        threshold: int = THRESHOLD,
    ) -> None:
        # Blanks dropped and order-preserving deduplication: a chain with the
        # same model twice would make one step of escalation do nothing, which
        # reads as the router being broken.
        seen: list[str] = []
        for key in chain:
            if key and key not in seen:
                seen.append(key)
        self.chain = seen
        self.enabled = enabled
        self.threshold = threshold

    @property
    def first(self) -> str:
        """Where a conversation starts. Empty only if the chain is."""
        return self.chain[0] if self.chain else ""

    @property
    def last(self) -> str:
        """The end of the chain: nothing to escalate to beyond this."""
        return self.chain[-1] if self.chain else ""

    def position(self, key: str | None) -> int:
        """Where `key` sits in the chain, or -1 if it is not in it.

        A model chosen by hand is usually not in the chain at all, and that
        has to mean "no floor" rather than "position 0" - otherwise picking a
        big model by hand would let the router demote the next turn.
        """
        try:
            return self.chain.index(key or "")
        except ValueError:
            return -1

    def choose(
        self,
        prompt: str,
        *,
        current_key: str | None = None,
        reached: str = "",
    ) -> RouteDecision:
        """Decide which model should handle `prompt`.

        `reached` is the furthest-along model this conversation has already
        used. Together with `current_key` it sets a floor, because routing
        down would pay the switch cost twice to give back RAM already spent.
        """
        if not self.enabled:
            return RouteDecision(
                key=current_key or self.first, reason="auto-routing off"
            )
        if not self.chain:
            return RouteDecision(key=current_key or "", reason="no models to route between")

        score, reasons = self.score(prompt)
        # One threshold of score per step along the chain. With two entries
        # this is the original rule exactly.
        step = max(1, self.threshold)
        wanted = min(score // step, len(self.chain) - 1)

        floor = max(self.position(current_key), self.position(reached))
        if floor > wanted:
            return RouteDecision(
                key=self.chain[floor],
                score=score,
                reason=(
                    f"staying on {self.chain[floor]} for this conversation"
                ),
            )

        if wanted == 0:
            return RouteDecision(
                key=self.chain[0], score=score, reason="looks simple enough"
            )
        return RouteDecision(
            key=self.chain[wanted],
            score=score,
            reason="looks involved: " + ", ".join(reasons),
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
