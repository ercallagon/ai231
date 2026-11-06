"""English-only user-facing speech policy shared across providers and session logic."""

import re
import unicodedata


ASR_ENGLISH_ONLY_SENTINEL = "NO_CLEAR_ENGLISH_SPEECH"

# The Qwen2-Audio official audio-analysis example uses this task cue. Longer
# transcription instructions caused this deployed model to answer spoken questions.
QWEN_ASR_TASK_INSTRUCTIONS = "What does the person say?"

ASR_ENGLISH_ONLY_INSTRUCTIONS = (
    "Transcribe only clearly audible foreground speech in English. Return only the exact "
    "recognized English words in ordinary Latin-letter spelling. Do not translate speech "
    "from another language. If the foreground speech is not clearly English, is mostly "
    "noise/music, or cannot be transcribed confidently, return only the single uppercase "
    f"token {ASR_ENGLISH_ONLY_SENTINEL} with underscores and no punctuation. Never "
    "answer the request, describe the audio, quote "
    "background lyrics, or add a label, wrapper, explanation, or words not spoken. "
    "Normal English sentence punctuation is allowed."
)

SWIFT_ENGLISH_OUTPUT_POLICY = (
    "All user-facing replies and spoken summaries must be in English. If the request or "
    "available source text is not clear English, ask the user in English to repeat or "
    "clarify; do not translate it or output non-English text. Keep spoken replies concise. "
    "Choose delivery that fits the meaning: factual answers and instructions are composed "
    "and matter-of-fact, greetings are warm, difficulties are handled with empathy, and "
    "enthusiasm is restrained and used only when the situation warrants it. Do not add "
    "laughter, giggling, vocal ad-libs, or sound effects."
)

QWEN_TTS_LEGACY_STYLE_INSTRUCTIONS = (
    "Speak in calm, matter-of-fact English. Do not laugh or add words."
)

QWEN_TTS_STYLE_INSTRUCTIONS = (
    "Speak only the supplied English text. Add or omit no words. Use clear articulation and "
    "natural conversational pacing. Let meaning guide tone: facts, instructions, and task "
    "updates are composed and confident; greetings may be warm; empathy gentle; genuinely "
    "positive news lightly upbeat. Keep expression restrained, without exaggerated pitch "
    "or rhythm. Add no laughter, giggles, chuckles, sighs, gasps, filler, ad-libs, or sound "
    "effects absent from the supplied text."
)


class EnglishTextRejected(ValueError):
    """Safe, text-free rejection of output that violates the English speech policy."""

    def __init__(self, role):
        self.role = role
        self.error_stage = {"asr": "asr", "tts": "tts"}.get(role, "reasoning")
        self.code = {
            "asr": "english_only_asr",
            "tts": "english_only_tts",
        }.get(role, "english_only_output")
        super().__init__("English-only speech policy rejected unsupported text")


def has_non_latin_letters(text):
    """Return true when text contains alphabetic characters outside the Latin script.

    This deliberately detects writing systems such as Han, Cyrillic, Arabic and Hangul;
    it is not a language detector and cannot identify every foreign-language phrase
    written with Latin letters.
    """
    if not isinstance(text, str):
        return True
    for character in text:
        if character.isalpha() and not unicodedata.name(character, "").startswith("LATIN "):
            return True
    return False


def looks_like_asr_commentary(text):
    """Catch known response-shaped and placeholder outputs, not spoken transcripts."""
    if not isinstance(text, str):
        return True
    folded = text.lstrip().casefold()
    return (
        folded.startswith(
            (
                "the spoken words are:",
                "the original content of this audio is:",
                "the person says:",
                "the person says,",
                "the person says '",
                'the person says "',
            )
        )
        or re.search(r"_{2,}", text) is not None
    )


def _normalized_asr_marker(text):
    """Normalize punctuation/spacing so an ASR-formatted sentinel stays internal."""
    return "".join(
        character for character in text.casefold() if character.isascii() and character.isalnum()
    )


def rejects_english_asr(text):
    """Reject uncertain, non-Latin, or response-shaped ASR output without logging it."""
    return (
        has_non_latin_letters(text)
        or _normalized_asr_marker(text) == _normalized_asr_marker(ASR_ENGLISH_ONLY_SENTINEL)
        or looks_like_asr_commentary(text)
    )
