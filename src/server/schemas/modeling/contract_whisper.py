from typing import Literal, Optional

from pydantic import BaseModel, Field


class OVGenAI_WhisperGenConfig(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio")
    language: Optional[str] = Field(None, description="Force the spoken language (ISO-639-1 code or name); None = detect")
    task: Literal["transcribe", "translate"] = Field("transcribe", description="translate = output English")
    return_timestamps: bool = Field(False, description="Segment start/end times")
    word_timestamps: bool = Field(False, description="Per-word start/end times")
