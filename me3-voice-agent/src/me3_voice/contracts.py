"""Bounded, local PCM contracts; no paths or provider credentials cross MCP."""

import base64
import binascii

from pydantic import BaseModel, Field

RATE = 16000
FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * 2
FRAME_SECONDS = FRAME_SAMPLES / RATE
MAX_AUDIO_BYTES = RATE * 2 * 30


class TurnKey(BaseModel):
    session_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    turn_id: int = Field(ge=1, strict=True)


def encode_pcm(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def decode_pcm(value: str) -> bytes:
    if len(value) > (FRAME_BYTES * 4 // 3 + 4):
        raise ValueError("chunk exceeds 512 mono PCM16 samples")
    try:
        pcm = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64 PCM") from exc
    if not pcm or len(pcm) > FRAME_BYTES or len(pcm) % 2:
        raise ValueError("expected 1..512 mono PCM16 samples")
    return pcm
