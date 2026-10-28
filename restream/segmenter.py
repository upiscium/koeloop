"""Incremental, lossless segmentation with optional conservative syntax gating."""

import time
from dataclasses import dataclass
from typing import ClassVar

from . import syntax


@dataclass(frozen=True)
class Segment:
    text: str
    reason: str


class Segmenter:
    MODES = frozenset({"raw", "sentence", "balanced", "syntax"})
    OPEN: ClassVar[dict[str, str]] = {"「": "」", "『": "』", "（": "）", "(": ")", "[": "]", "【": "】", "“": "”"}
    CLOSE = frozenset(OPEN.values())
    END = frozenset("。！？!?\n")
    CLAUSE = frozenset("、，,；;")

    def __init__(self, mode="balanced", min_chars=24, max_chars=160, lookahead=12, analyzer=None):
        if mode not in self.MODES:
            raise ValueError("unknown segmentation mode")
        if not (1 <= min_chars <= max_chars <= 1000 and 0 <= lookahead <= max_chars):
            raise ValueError("invalid segmentation limits")
        self.mode = mode
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.lookahead = lookahead
        self.buffer = ""
        self.effective_mode = mode
        self.analyzer = analyzer if analyzer is not None else syntax.ANALYZER
        self.analysis_calls = 0
        self.analysis_ms = 0.0
        self.fallback = None
        self.notices = []
        self.diagnostics = []
        self._last_analysis_length = 0
        if mode == "syntax" and analyzer is None and not syntax.installed():
            self._fallback("日本語解析器が未導入のため、文末方式で処理します。nix develop または syntax extra を利用してください。")

    def _fallback(self, message):
        self.effective_mode = "sentence"
        self.fallback = message
        self.notices.append(message)

    def metrics(self):
        return {"requested_mode": self.mode, "effective_mode": self.effective_mode,
                "analysis_calls": self.analysis_calls, "analysis_ms": round(self.analysis_ms, 2),
                "fallback": self.fallback}

    def _syntax_boundary(self, candidates):
        # Require two different views, each with at least the configured lookahead.
        # Eight extra characters test whether the same dependency stays plausible.
        lookahead = max(12, self.lookahead)
        view = self.buffer[:self.max_chars + lookahead + 8]
        candidates = [end for end in candidates if len(view) - end >= lookahead + 8]
        if not candidates or len(view) - self._last_analysis_length < 8:
            return None
        self._last_analysis_length = len(view)
        started = time.monotonic()
        try:
            self.analysis_calls += 1
            earlier = self.analyzer.parse(view[:-8])
            self.analysis_calls += 1
            later = self.analyzer.parse(view)
            approved = None
            decisions = []
            for end in candidates:
                first = syntax.assess(earlier, end)
                second = syntax.assess(later, end)
                stable = first.accepted and second.accepted and first.signature == second.signature
                reason = ("stable_clause" if stable else second.reason if not second.accepted
                          else first.reason if not first.accepted else "unstable_dependency")
                decisions.append({"offset": end, "accepted": stable, "reason": reason,
                                  "preview": view[max(0, end - 24):end], "context_chars": len(view)})
                if stable:
                    approved = end
            self.diagnostics.extend({**decision, "selected": decision["offset"] == approved}
                                    for decision in decisions)
            return (approved, "syntax_clause") if approved is not None else None
        except Exception:  # noqa: BLE001 - optional parser failure must not corrupt text or silently use comma splitting.
            self._fallback("構文解析に失敗したため、この発話セッションは文末方式で続行します。")
            return None
        finally:
            self.analysis_ms += (time.monotonic() - started) * 1000

    def feed(self, text: str) -> list[Segment]:
        if self.mode == "raw":
            # Even the baseline has a safety limit on a single synthesis request.
            return [Segment(text[i:i + self.max_chars], "raw" if len(text) <= self.max_chars else "forced_limit")
                    for i in range(0, len(text), self.max_chars)]
        self.buffer += text
        return self._drain()

    def flush(self) -> list[Segment]:
        return self._drain(final=True)

    def idle(self) -> list[Segment]:
        """Release a complete sentence at a stream pause, never an arbitrary fragment."""
        return self._drain(idle=True)

    def _boundary(self, final: bool, idle: bool):
        text = self.buffer
        stack: list[str] = []
        clause = None
        candidates = []
        terminal = False
        for i, char in enumerate(text[:self.max_chars]):
            if char in self.OPEN:
                stack.append(self.OPEN[char])
            elif stack and char == stack[-1]:
                stack.pop()
            # ASCII periods are intentionally not split: decimals, URLs, abbreviations.
            if char in self.END:
                terminal = True
            elif char not in self.CLOSE:
                terminal = False
            end = i + 1
            if terminal and not stack:
                # Wait for following punctuation/closers, unless the source pauses/ends.
                if end < len(text) and text[end] in self.CLOSE | self.END:
                    continue
                if end < len(text) or final or idle or end == self.max_chars:
                    return end, "sentence"
                return None
            # Do not break a digit grouping such as 1,000.
            digit_comma = (char == "," and i and text[i - 1].isdigit()
                           and end < len(text) and text[end].isdigit())
            if (self.effective_mode in {"balanced", "syntax"} and not stack and char in self.CLAUSE
                    and end >= self.min_chars and len(text) - end >= self.lookahead and not digit_comma):
                clause = end
                candidates.append(end)
        if clause is not None and self.effective_mode == "balanced":
            return clause, "clause"
        if candidates and self.effective_mode == "syntax":
            boundary = self._syntax_boundary(candidates)
            if boundary is not None:
                return boundary
        if len(text) >= self.max_chars:
            return self.max_chars, "forced_limit"
        if final and text:
            return len(text), "end_of_input"
        return None

    def _drain(self, final=False, idle=False) -> list[Segment]:
        result = []
        while self.buffer:
            boundary = self._boundary(final, idle)
            if boundary is None:
                break
            end, reason = boundary
            result.append(Segment(self.buffer[:end], reason))
            self.buffer = self.buffer[end:]
            self._last_analysis_length = 0
        return result
