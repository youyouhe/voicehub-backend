#!/usr/bin/env python3
"""
VoiceHub Backend - Enhanced TTS API for VoiceHub frontend

A lightweight backend server that provides TTS functionality using CosyVoice.
Supports multiple inference modes: SFT, Zero-Shot, Cross-Lingual, Instruct2
"""

import os
import sys
import io
import base64
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Add CosyVoice to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "CosyVoice"))
sys.path.insert(0, str(ROOT_DIR / "CosyVoice" / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.common import set_all_random_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9880
DEFAULT_MODEL_DIR = "CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"


# =============================================================================
# Request/Response Models
# =============================================================================
class TTSRequest(BaseModel):
    """TTS generation request with mode selection."""
    mode: str = Field(default="zero_shot", description="Inference mode: sft, zero_shot, cross_lingual, instruct2")
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")
    seed: Optional[int] = Field(None, ge=0, description="Random seed")

    # SFT mode
    speaker_id: Optional[str] = Field(None, description="Speaker ID for SFT/Instruct2 mode")

    # Zero-shot & Cross-lingual mode
    prompt_text: Optional[str] = Field(None, description="Text spoken in reference audio")
    prompt_audio: Optional[str] = Field(None, description="Base64-encoded reference audio")

    # Instruct2 mode
    instruct_text: Optional[str] = Field(None, description="Instruction text")


class TTSResponse(BaseModel):
    """TTS generation response."""
    success: bool
    audio_data: str  # base64 encoded
    sample_rate: int
    duration: float
    mode: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    is_model_loaded: bool = Field(alias="model_loaded", default=True)
    cosyvoice_version: Optional[str] = Field(alias="model_version", default=None)
    speakers_count: int = 0
    available_modes: list = []

    class Config:
        populate_by_name = True


class SpeakersResponse(BaseModel):
    """Available speakers response."""
    speakers: list
    count: int
    mode: str


# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(
    title="VoiceHub Backend",
    description="Enhanced TTS API for VoiceHub frontend using CosyVoice3",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
cosyvoice_model = None


def get_model():
    """Get or initialize the CosyVoice model."""
    global cosyvoice_model
    if cosyvoice_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return cosyvoice_model


def decode_base64_audio(audio_base64: str, target_sr: int = 16000) -> torch.Tensor:
    """Decode base64-encoded audio to tensor."""
    try:
        audio_bytes = base64.b64decode(audio_base64)
        with io.BytesIO(audio_bytes) as audio_buffer:
            speech, sample_rate = torchaudio.load(audio_buffer, backend='soundfile')
        speech = speech.mean(dim=0, keepdim=True)
        if sample_rate != target_sr:
            if sample_rate < 16000:
                raise ValueError(f"Audio sample rate {sample_rate} is too low (minimum 16kHz)")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            speech = resampler(speech)
        return speech
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {str(e)}")


def generate_audio_bytes(model_output):
    """Convert model output to audio bytes."""
    for output in model_output:
        audio_tensor = output['tts_speech']
        audio_bytes = (audio_tensor.numpy() * (2 ** 15)).astype(np.int16).tobytes()
        yield audio_bytes


# =============================================================================
# API Endpoints
# =============================================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model = get_model()
    model_version = model.__class__.__name__.replace("CosyVoice", "")
    speakers = model.list_available_spks()

    # Determine available modes based on model version
    available_modes = ["zero_shot", "cross_lingual"]
    if len(speakers) > 0:
        available_modes.insert(0, "sft")
    if model_version in ["2", "3"]:
        available_modes.append("instruct2")

    return HealthResponse(
        status="healthy",
        is_model_loaded=True,
        cosyvoice_version=model_version,
        speakers_count=len(speakers),
        available_modes=available_modes
    )


@app.get("/speakers", response_model=SpeakersResponse)
async def list_speakers():
    """List available speaker IDs (for SFT mode)."""
    model = get_model()
    speakers = model.list_available_spks()
    return SpeakersResponse(
        speakers=speakers,
        count=len(speakers),
        mode="sft"
    )


@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    Generate speech from text using CosyVoice.

    Supports multiple modes:
    - **sft**: Pretrained speaker (requires speaker_id)
    - **zero_shot**: Voice cloning (requires prompt_text + prompt_audio)
    - **cross_lingual**: Cross-language synthesis (requires prompt_audio)
    - **instruct2**: Natural language control (requires speaker_id + instruct_text + optional prompt_audio)
    """
    model = get_model()
    mode = request.mode

    # Set random seed if provided
    if request.seed is not None:
        set_all_random_seed(request.seed)

    start_time = time.time()

    try:
        if mode == "sft":
            # SFT mode: use pretrained speaker
            if not request.speaker_id:
                raise HTTPException(status_code=400, detail="speaker_id required for SFT mode")
            if len(model.list_available_spks()) == 0:
                raise HTTPException(status_code=400, detail="Model has no pretrained speakers. Use zero_shot mode instead.")

            model_output = model.inference_sft(
                tts_text=request.text,
                spk_id=request.speaker_id,
                stream=False,
                speed=request.speed
            )

        elif mode == "zero_shot":
            # Zero-shot mode: clone voice from reference audio
            if not request.prompt_text:
                raise HTTPException(status_code=400, detail="prompt_text required for zero_shot mode")
            if not request.prompt_audio:
                raise HTTPException(status_code=400, detail="prompt_audio required for zero_shot mode")

            prompt_speech = decode_base64_audio(request.prompt_audio)
            model_output = model.inference_zero_shot(
                tts_text=request.text,
                prompt_text=request.prompt_text,
                prompt_wav=prompt_speech,
                stream=False,
                speed=request.speed
            )

        elif mode == "cross_lingual":
            # Cross-lingual mode: synthesize in different language
            if not request.prompt_audio:
                raise HTTPException(status_code=400, detail="prompt_audio required for cross_lingual mode")

            prompt_speech = decode_base64_audio(request.prompt_audio)
            model_output = model.inference_cross_lingual(
                tts_text=request.text,
                prompt_wav=prompt_speech,
                stream=False,
                speed=request.speed
            )

        elif mode == "instruct2":
            # Instruct2 mode: natural language control
            if not request.instruct_text:
                raise HTTPException(status_code=400, detail="instruct_text required for instruct2 mode")
            if not request.speaker_id:
                raise HTTPException(status_code=400, detail="speaker_id required for instruct2 mode")

            model_output = model.inference_instruct2(
                tts_text=request.text,
                instruct_text=request.instruct_text,
                prompt_wav=None,
                stream=False,
                speed=request.speed
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")

        # Collect audio chunks
        audio_chunks = []
        for output in model_output:
            audio = output['tts_speech']
            audio_chunks.append(audio.numpy().flatten())

        # Combine and convert to bytes
        combined_audio = np.concatenate(audio_chunks)
        audio_int16 = (combined_audio * (2 ** 15)).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # Calculate duration
        duration = len(audio_int16) / model.sample_rate

        # Encode to base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        processing_time = time.time() - start_time
        logger.info(f"Generated {duration:.2f}s audio in {processing_time:.2f}s using {mode} mode")

        return TTSResponse(
            success=True,
            audio_data=audio_base64,
            sample_rate=model.sample_rate,
            duration=duration,
            mode=mode
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


# =============================================================================
# Main
# =============================================================================
def load_model(model_dir: str):
    """Load the CosyVoice model."""
    global cosyvoice_model
    logger.info(f"Loading CosyVoice model from: {model_dir}")

    try:
        cosyvoice_model = AutoModel(model_dir=model_dir)
        logger.info(f"Model loaded successfully. Sample rate: {cosyvoice_model.sample_rate}Hz")
        logger.info(f"Available speakers: {len(cosyvoice_model.list_available_spks())}")
        logger.info(f"Model version: {cosyvoice_model.__class__.__name__}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="VoiceHub Backend - Enhanced TTS API")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR, help="CosyVoice model directory")
    args = parser.parse_args()

    # Load model
    if not load_model(args.model_dir):
        logger.error("Failed to load model. Exiting.")
        sys.exit(1)

    # Run server
    logger.info(f"Starting VoiceHub Backend on {args.host}:{args.port}")
    logger.info(f"API docs available at http://{args.host}:{args.port}/docs")
    logger.info(f"Supported modes: zero_shot, cross_lingual, instruct2")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
