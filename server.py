#!/usr/bin/env python3
"""
VoiceHub Backend - Minimal TTS API for VoiceHub frontend

A lightweight backend server that provides TTS functionality using CosyVoice.
Designed specifically for the VoiceHub frontend application.
"""

import os
import sys
import io
import base64
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

# Add CosyVoice to path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "CosyVoice"))
sys.path.insert(0, str(ROOT_DIR / "CosyVoice" / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.file_utils import load_wav

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
    """TTS generation request."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    speaker_id: str = Field(default="中文女", description="Speaker ID for SFT mode")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed")
    seed: Optional[int] = Field(None, ge=0, description="Random seed for reproducibility")


class TTSResponse(BaseModel):
    """TTS generation response."""
    success: bool
    audio_data: str  # base64 encoded
    sample_rate: int
    duration: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    speakers_count: int = 0


# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(
    title="VoiceHub Backend",
    description="Minimal TTS API for VoiceHub frontend using CosyVoice",
    version="1.0.0"
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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = cosyvoice_model is not None
    model_version = None
    speakers_count = 0

    if model_loaded:
        try:
            model_version = cosyvoice_model.__class__.__name__.replace("CosyVoice", "")
            speakers = cosyvoice_model.list_available_spks()
            speakers_count = len(speakers)
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_version=model_version,
        speakers_count=speakers_count
    )


@app.get("/speakers")
async def list_speakers():
    """List available speaker IDs."""
    model = get_model()
    speakers = model.list_available_spks()
    return {"speakers": speakers, "count": len(speakers)}


@app.post("/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    Generate speech from text using CosyVoice.

    Supports SFT (Supervised Fine-Tuning) mode with pretrained speakers.
    """
    model = get_model()

    # Set random seed if provided
    if request.seed is not None:
        torch.manual_seed(request.seed)
        np.random.seed(request.seed)

    try:
        start_time = time.time()

        # Generate audio using SFT mode
        model_output = model.inference_sft(
            tts_text=request.text,
            spk_id=request.speaker_id,
            stream=False,
            speed=request.speed
        )

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
        logger.info(f"Generated {duration:.2f}s audio in {processing_time:.2f}s")

        return TTSResponse(
            success=True,
            audio_data=audio_base64,
            sample_rate=model.sample_rate,
            duration=duration
        )

    except Exception as e:
        logger.error(f"TTS generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


def load_model(model_dir: str):
    """Load the CosyVoice model."""
    global cosyvoice_model
    logger.info(f"Loading CosyVoice model from: {model_dir}")

    try:
        cosyvoice_model = AutoModel(model_dir=model_dir)
        logger.info(f"Model loaded successfully. Sample rate: {cosyvoice_model.sample_rate}Hz")
        logger.info(f"Available speakers: {len(cosyvoice_model.list_available_spks())}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="VoiceHub Backend - Minimal TTS API")
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

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
