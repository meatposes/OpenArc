from typing import Literal, Optional

from pydantic import BaseModel, Field


class OV_DiarizationGenConfig(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio file (any format libsndfile reads)")
    task: Literal["diarize", "voiceprint"] = Field(
        "diarize",
        description="diarize: speaker turns plus one embedding per speaker. "
                    "voiceprint: one embedding for audio that holds a single speaker's speech.",
    )
    num_speakers: Optional[int] = Field(None, ge=1, description="Exact number of speakers, when known")
    min_speakers: Optional[int] = Field(None, ge=1, description="Lower bound on the number of speakers")
    max_speakers: Optional[int] = Field(None, ge=1, description="Upper bound on the number of speakers")
