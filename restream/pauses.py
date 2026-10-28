"""Adjust VOICEVOX pause durations without changing text or speech phonemes.

Sentence-final punctuation is normally discarded by audio_query. Append exact
PCM silence after synthesis, instead of asking the vocoder to regenerate a
longer tail or sleeping in the browser. This keeps speech samples unchanged.
"""

import io
import wave
from dataclasses import dataclass

SENTENCE = frozenset("。！？!?\n")
COMMA = frozenset("、，,；;")
CLOSERS = frozenset("」』）)]】”\"'〉》")


@dataclass(frozen=True)
class PauseSettings:
    sentence_extra_ms: int = 200
    comma_scale: float = 0.8

    def __post_init__(self):
        if not 0 <= self.sentence_extra_ms <= 1000 or not 0 <= self.comma_scale <= 2:
            raise ValueError("invalid pause settings")


def ending(text: str) -> tuple[str | None, str]:
    """Return terminal boundary kind and the remaining text before punctuation."""
    end = len(text)
    newline = False
    while end and (text[end - 1].isspace() or text[end - 1] in CLOSERS):
        newline |= text[end - 1] == "\n"
        end -= 1
    start = end
    while start and text[start - 1] in SENTENCE | COMMA:
        start -= 1
    tail = text[start:end]
    if any(char in SENTENCE for char in tail) or newline:
        return "sentence", text[:start]
    if any(char in COMMA for char in tail):
        return "comma", text[:start]
    return None, text[:end]


def comma_groups(text: str) -> int:
    groups = 0
    in_group = False
    for index, char in enumerate(text):
        numeric = (char == "," and index > 0 and index + 1 < len(text)
                   and text[index - 1].isdigit() and text[index + 1].isdigit())
        comma = char in COMMA and not numeric
        if comma and not in_group:
            groups += 1
        in_group = comma
    return groups


def apply_pauses(query: dict, text: str, settings: PauseSettings) -> dict:
    """Modify a fresh audio_query in place; return non-sensitive adjustment details.

Internal sentence pauses cannot be reliably distinguished from comma pauses in
VOICEVOX's kana output. Preserve them all when punctuation mapping is ambiguous.
    Sentence silence is added separately by append_sentence_silence. No PCM
    trimming or changes to pitch, pronunciation, or consonants are performed.
"""
    kind, interior = ending(text)
    info = {"ending": kind, "sentence_extra_ms": 0, "comma_scale": settings.comma_scale,
            "internal": "unchanged", "adjusted_pause_moras": 0}
    if kind == "sentence" and settings.sentence_extra_ms:
        info["sentence_extra_ms"] = settings.sentence_extra_ms
    elif kind == "comma" and settings.comma_scale != 1:
        query["postPhonemeLength"] = query.get("postPhonemeLength", 0.1) * settings.comma_scale

    if settings.comma_scale == 1:
        return info
    phrases = query.get("accent_phrases", [])
    pauses = [(index, phrase["pause_mora"]) for index, phrase in enumerate(phrases)
              if phrase.get("pause_mora") is not None]
    terminal_index = len(phrases) - 1 if kind else None
    internal = [(index, pause) for index, pause in pauses if index != terminal_index]
    selected = set()
    if any(char in SENTENCE for char in interior):
        info["internal"] = "preserved_mixed_punctuation"
    elif comma_groups(interior) != len(internal):
        info["internal"] = "preserved_unmapped_pauses"
    else:
        selected.update(index for index, _ in internal)
        if internal:
            info["internal"] = "comma_scaled"
    # Some Engine variants keep terminal pause_mora; don't double-preserve its comma pause.
    if kind == "comma":
        selected.update(index for index, _ in pauses if index == terminal_index)
    if selected:
        # Bake global overrides into every existing pause before normalizing them,
        # so non-selected pauses (e.g. terminal sentence pauses) keep their durations.
        override = query.get("pauseLength")
        scale = query.get("pauseLengthScale", 1.0)
        for index, pause in pauses:
            duration = (override if override is not None else pause["vowel_length"]) * scale
            pause["vowel_length"] = duration * (settings.comma_scale if index in selected else 1)
        if "pauseLength" in query:
            query["pauseLength"] = None
        if "pauseLengthScale" in query:
            query["pauseLengthScale"] = 1.0
        info["adjusted_pause_moras"] = len(selected)
    return info


def append_sentence_silence(audio: bytes, text: str, settings: PauseSettings) -> bytes:
    """Append a speed-independent gap; keep all original PCM samples intact."""
    if not settings.sentence_extra_ms or ending(text)[0] != "sentence":
        return audio
    with wave.open(io.BytesIO(audio), "rb") as source:
        parameters = source.getparams()
        if parameters.sampwidth != 2 or parameters.nchannels != 1 or parameters.framerate != 24000:
            raise ValueError("pause adjustment requires 24 kHz mono 16-bit PCM")
        frames = source.readframes(source.getnframes())
        if len(frames) != parameters.nframes * 2:
            raise ValueError("truncated PCM WAV")
    extra_frames = round(parameters.framerate * settings.sentence_extra_ms / 1000)
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setparams(parameters)
        target.writeframes(frames + b"\x00\x00" * extra_frames)
    return output.getvalue()
