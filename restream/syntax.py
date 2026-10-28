"""Conservative clause-boundary assessment using real Japanese dependency parses.

The parser proposes a tree, not a proof of semantic completeness. Only marked
predicate clauses are eligible; nominal phrases and unresolved relations wait.
"""

from dataclasses import dataclass
from importlib.util import find_spec
from threading import Lock

MODEL = "ja_core_news_sm"


@dataclass(frozen=True)
class Token:
    start: int
    end: int
    text: str
    pos: str
    tag: str
    dep: str
    head: int


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str
    signature: tuple = ()


def installed() -> bool:
    return all(find_spec(name) is not None for name in ("spacy", MODEL, "sudachipy", "sudachidict_core"))


class SpacyAnalyzer:
    """One CPU model per process; serialize access to the tokenizer/model."""

    def __init__(self):
        self._lock = Lock()
        self._nlp = None

    def parse(self, text: str) -> tuple[Token, ...]:
        with self._lock:
            if self._nlp is None:
                import spacy

                spacy.require_cpu()
                # No download call: the model must already be installed.
                self._nlp = spacy.load(MODEL, exclude=["ner"])
            # Only immutable Python values leave this zone, never spaCy Doc/Token objects.
            # Avoid retaining an ever-growing vocabulary of users' streamed text.
            with self._nlp.memory_zone():
                doc = self._nlp(text)
                if not doc.has_annotation("DEP"):
                    raise RuntimeError("dependency annotation is unavailable")
                return tuple(Token(t.idx, t.idx + len(t.text), t.text, t.pos_, t.tag_, t.dep_, t.head.i)
                             for t in doc)


ANALYZER = SpacyAnalyzer()


def predicate(tokens: tuple[Token, ...], index: int) -> bool:
    token = tokens[index]
    return token.pos in {"VERB", "ADJ"} or (
        token.pos in {"NOUN", "PRON"} and any(t.head == index and t.dep == "cop" for t in tokens))


def assess(tokens: tuple[Token, ...], offset: int) -> Decision:
    """offset is the first character *after* the candidate comma."""
    punctuation = next((i for i, t in enumerate(tokens) if t.end == offset and t.pos == "PUNCT"), None)
    if punctuation is None:
        return Decision(False, "not_token_boundary")
    comma = tokens[punctuation]
    head_index = comma.head
    head = tokens[head_index]
    if head.start >= comma.start or not predicate(tokens, head_index):
        return Decision(False, "not_predicate")
    if head.dep == "acl":
        return Decision(False, "relative_clause")
    if head.dep != "advcl":
        return Decision(False, "not_clause")

    tail = [t for t in tokens[head_index + 1:punctuation] if t.pos not in {"SPACE", "PUNCT"}]
    # A bare past-tense ending may modify a future noun, even if mislabelled advcl.
    # Require a grammatical connective, not just a punctuation-attached verb.
    tail_text = "".join(t.text for t in tail)
    marked = tail and (
        tail[-1].pos == "SCONJ" and tail[-1].dep in {"mark", "fixed"}
        or tail_text.endswith(("ので", "のに", "から", "けれど", "けれども", "けど", "ながら", "つつ"))
    )
    if not marked:
        return Decision(False, "no_connective")
    parent = tokens[head.head]
    if parent.start < offset or not predicate(tokens, head.head):
        return Decision(False, "right_predicate_pending")

    # Reject a clause whose enclosing dependency chain still modifies a noun.
    visited = {head_index}
    cursor = head.head
    while cursor not in visited:
        visited.add(cursor)
        ancestor = tokens[cursor]
        if ancestor.dep == "acl":
            return Decision(False, "relative_clause")
        cursor = ancestor.head

    # A completed subordinate clause can attach to its following predicate.
    # Any OTHER crossing edge leaves an argument/phrase straddling the cut.
    for index, token in enumerate(tokens):
        if token.pos in {"SPACE", "PUNCT"}:
            continue
        if (token.start < offset) != (tokens[token.head].start < offset) and index != head_index:
            return Decision(False, "crossing_dependency")
    return Decision(True, "predicate_clause", (head.start, head.dep, parent.start))
