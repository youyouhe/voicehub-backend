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
import psutil
import subprocess
import json
from datetime import datetime
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
DEFAULT_MODEL_DIR = "CosyVoice/pretrained_models/CosyVoice2-0.5B"


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
    model_name: Optional[str] = Field(alias="model_name", default=None)
    speakers_count: int = 0
    available_modes: list = []
    uptime_seconds: Optional[int] = Field(alias="uptime_seconds", default=None)
    server_time: Optional[str] = Field(alias="server_time", default=None)

    class Config:
        populate_by_name = True


class SpeakersResponse(BaseModel):
    """Available speakers response."""
    speakers: list
    count: int
    mode: str


class CreateSpeakerRequest(BaseModel):
    """Create custom speaker request."""
    speaker_id: str = Field(..., description="Unique ID for the speaker")
    prompt_text: str = Field(..., description="Text spoken in reference audio")
    prompt_audio: str = Field(..., description="Base64-encoded reference audio")


class CreateSpeakerResponse(BaseModel):
    """Create speaker response."""
    success: bool
    speaker_id: str
    message: str


class GPUMetrics(BaseModel):
    """GPU metrics."""
    available: bool
    name: Optional[str] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    vram_total_mb: Optional[int] = None
    vram_used_mb: Optional[int] = None
    vram_free_mb: Optional[int] = None
    gpu_utilization_percent: Optional[float] = None
    temperature_celsius: Optional[int] = None
    power_draw_watts: Optional[float] = None
    error: Optional[str] = None


class CPUMetrics(BaseModel):
    """CPU metrics."""
    usage_percent: float
    cores: int


class MemoryMetrics(BaseModel):
    """Memory metrics."""
    total_gb: float
    used_gb: float
    available_gb: float
    usage_percent: float


class DiskMetrics(BaseModel):
    """Disk metrics."""
    total_gb: float
    used_gb: float
    available_gb: float
    usage_percent: float


class NetworkMetrics(BaseModel):
    """Network metrics."""
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int


class SystemMetricsResponse(BaseModel):
    """System metrics response."""
    cpu: CPUMetrics
    memory: MemoryMetrics
    gpu: GPUMetrics
    disk: DiskMetrics
    network: NetworkMetrics
    timestamp: str


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
# Server start time for uptime tracking
SERVER_START_TIME = time.time()
# Model directory for model name display
MODEL_DIR = DEFAULT_MODEL_DIR


def get_model():
    """Get or initialize the CosyVoice model."""
    global cosyvoice_model
    if cosyvoice_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return cosyvoice_model


import tempfile


def decode_base64_audio(audio_base64: str) -> str:
    """Decode base64-encoded audio to temporary file path."""
    try:
        audio_bytes = base64.b64decode(audio_base64)
        # Create temporary wav file
        fd, path = tempfile.mkstemp(suffix='.wav')
        with os.fdopen(fd, 'wb') as f:
            f.write(audio_bytes)
        return path
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

    # Extract model name from model directory
    model_name = os.path.basename(MODEL_DIR) if MODEL_DIR else None

    # Determine available modes based on model version
    available_modes = ["zero_shot", "cross_lingual"]
    if len(speakers) > 0:
        available_modes.insert(0, "sft")
    if model_version in ["2", "3"]:
        available_modes.append("instruct2")

    # Calculate uptime
    uptime_seconds = int(time.time() - SERVER_START_TIME)

    # Current server time
    server_time = datetime.utcnow().isoformat() + "Z"

    return HealthResponse(
        status="healthy",
        is_model_loaded=True,
        cosyvoice_version=model_version,
        model_name=model_name,
        speakers_count=len(speakers),
        available_modes=available_modes,
        uptime_seconds=uptime_seconds,
        server_time=server_time
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


@app.get("/speakers/{speaker_id}")
async def get_speaker(speaker_id: str):
    """
    Get details of a specific speaker.
    Returns speaker metadata including prompt_text (for reference).
    """
    model = get_model()
    speakers = model.list_available_spks()

    if speaker_id not in speakers:
        raise HTTPException(status_code=404, detail=f"Speaker '{speaker_id}' not found")

    # Get speaker info from spk2info
    spk_info = None
    if hasattr(model.frontend, 'spk2info') and speaker_id in model.frontend.spk2info:
        spk_info = model.frontend.spk2info[speaker_id]

    # Try to extract prompt_text from spk_info
    prompt_text = None
    if spk_info and 'prompt_text' in spk_info:
        # Decode token back to text
        prompt_text_tokens = spk_info['prompt_text']
        if hasattr(model.frontend, 'tokenizer'):
            try:
                prompt_text = model.frontend.tokenizer.decode(prompt_text_tokens[0].tolist())
            except:
                prompt_text = None

    return {
        "speaker_id": speaker_id,
        "prompt_text": prompt_text,
        "is_builtin": speaker_id in model.frontend.spk2info if hasattr(model.frontend, 'spk2info') else False
    }


def get_gpu_metrics() -> GPUMetrics:
    """Get GPU metrics using nvidia-smi."""
    try:
        # Use nvidia-smi to get GPU information (first GPU only)
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return GPUMetrics(available=False, error="nvidia-smi command failed")

        # Parse CSV output (may have multiple GPUs, use first one)
        lines = result.stdout.strip().split('\n')
        if not lines or not lines[0]:
            return GPUMetrics(available=False, error="No GPU data found")

        # Use first GPU
        values = lines[0].strip().split(', ')
        if len(values) < 7:
            return GPUMetrics(available=False, error="Invalid nvidia-smi output")

        return GPUMetrics(
            available=True,
            name=values[0].strip(),
            driver_version=values[1].strip(),
            cuda_version=None,  # Not available in query
            vram_total_mb=int(values[2].strip()),
            vram_used_mb=int(values[3].strip()),
            vram_free_mb=int(values[2].strip()) - int(values[3].strip()),
            gpu_utilization_percent=float(values[4].strip()),
            temperature_celsius=int(values[5].strip()),
            power_draw_watts=float(values[6].strip())
        )
    except FileNotFoundError:
        return GPUMetrics(available=False, error="nvidia-smi not found")
    except Exception as e:
        return GPUMetrics(available=False, error=str(e))


def get_cpu_metrics() -> CPUMetrics:
    """Get CPU metrics using psutil."""
    return CPUMetrics(
        usage_percent=psutil.cpu_percent(interval=0.1),
        cores=psutil.cpu_count()
    )


def get_memory_metrics() -> MemoryMetrics:
    """Get memory metrics using psutil."""
    mem = psutil.virtual_memory()
    return MemoryMetrics(
        total_gb=round(mem.total / (1024**3), 2),
        used_gb=round(mem.used / (1024**3), 2),
        available_gb=round(mem.available / (1024**3), 2),
        usage_percent=mem.percent
    )


def get_disk_metrics() -> DiskMetrics:
    """Get disk metrics using psutil."""
    disk = psutil.disk_usage('/')
    return DiskMetrics(
        total_gb=round(disk.total / (1024**3), 2),
        used_gb=round(disk.used / (1024**3), 2),
        available_gb=round(disk.free / (1024**3), 2),
        usage_percent=round(disk.percent, 2)
    )


def get_network_metrics() -> NetworkMetrics:
    """Get network metrics using psutil."""
    net = psutil.net_io_counters()
    return NetworkMetrics(
        bytes_sent=net.bytes_sent,
        bytes_recv=net.bytes_recv,
        packets_sent=net.packets_sent,
        packets_recv=net.packets_recv
    )


@app.get("/system/metrics", response_model=SystemMetricsResponse)
async def system_metrics():
    """
    Get system resource metrics including CPU, memory, GPU, disk, and network.

    Returns comprehensive system status for frontend monitoring dashboards.
    """
    return SystemMetricsResponse(
        cpu=get_cpu_metrics(),
        memory=get_memory_metrics(),
        gpu=get_gpu_metrics(),
        disk=get_disk_metrics(),
        network=get_network_metrics(),
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@app.post("/speakers", response_model=CreateSpeakerResponse)
async def create_speaker(request: CreateSpeakerRequest):
    """
    Create a custom speaker using zero-shot voice cloning.

    The speaker can be used with instruct2 mode for emotion/style control.
    """
    model = get_model()

    # Check if speaker_id already exists
    existing_speakers = model.list_available_spks()
    if request.speaker_id in existing_speakers:
        raise HTTPException(status_code=400, detail=f"Speaker '{request.speaker_id}' already exists")

    # Decode audio to temporary file
    temp_audio_path = None
    try:
        temp_audio_path = decode_base64_audio(request.prompt_audio)

        # Add zero-shot speaker
        success = model.add_zero_shot_spk(
            request.prompt_text,
            temp_audio_path,
            request.speaker_id
        )

        if success:
            # Persist speaker info to disk
            # Keep all fields (prompt_text, prompt_speech_feat, embeddings, etc.)
            # This allows zero-shot mode to reuse the saved speaker without re-uploading audio
            model.save_spkinfo()
            logger.info(f"Created custom speaker: {request.speaker_id}")
            return CreateSpeakerResponse(
                success=True,
                speaker_id=request.speaker_id,
                message=f"Speaker '{request.speaker_id}' created successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to create speaker")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create speaker: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create speaker: {str(e)}")
    finally:
        # Clean up temp file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass


@app.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str):
    """Delete a custom speaker by ID."""
    model = get_model()

    # Check if speaker exists
    existing_speakers = model.list_available_spks()
    if speaker_id not in existing_speakers:
        raise HTTPException(status_code=404, detail=f"Speaker '{speaker_id}' not found")

    try:
        # Remove speaker from model's speaker info
        # Check both model.spk2info and model.frontend.spk2info
        spk2info = None
        if hasattr(model, 'spk2info') and speaker_id in model.spk2info:
            spk2info = model.spk2info
        elif hasattr(model.frontend, 'spk2info') and speaker_id in model.frontend.spk2info:
            spk2info = model.frontend.spk2info

        if spk2info:
            del spk2info[speaker_id]
            # Persist changes to disk
            model.save_spkinfo()
            logger.info(f"Deleted speaker: {speaker_id}")
            return {"success": True, "message": f"Speaker '{speaker_id}' deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Cannot delete built-in speakers")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete speaker: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete speaker: {str(e)}")


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
    # Debug: Log full request
    logger.info("=" * 60)
    logger.info("TTS Request received:")
    logger.info(f"  mode: {request.mode}")
    logger.info(f"  text: {request.text[:100]}{'...' if len(request.text) > 100 else ''}")
    logger.info(f"  text_length: {len(request.text)} characters")
    logger.info(f"  speed: {request.speed}")
    logger.info(f"  seed: {request.seed}")
    logger.info(f"  speaker_id: {request.speaker_id}")
    logger.info(f"  instruct_text: {request.instruct_text[:100] if request.instruct_text else None}{'...' if request.instruct_text and len(request.instruct_text) > 100 else ''}")
    logger.info(f"  prompt_text: {request.prompt_text[:100] if request.prompt_text else None}{'...' if request.prompt_text and len(request.prompt_text) > 100 else ''}")
    logger.info(f"  prompt_audio_length: {len(request.prompt_audio) if request.prompt_audio else 0} bytes")
    logger.info("=" * 60)

    model = get_model()
    mode = request.mode

    # Set random seed if provided
    if request.seed is not None:
        set_all_random_seed(request.seed)

    start_time = time.time()

    # Track temporary files for cleanup
    temp_files = []

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
            # Supports two modes:
            # 1. Traditional: provide prompt_text and prompt_audio
            # 2. Using saved speaker: provide speaker_id (no need to upload audio each time)

            if request.speaker_id:
                # Mode 2: Use saved speaker (similar to instruct2 but without instruct_text)
                model_output = model.inference_zero_shot(
                    tts_text=request.text,
                    prompt_text="",
                    prompt_wav="CosyVoice/asset/zero_shot_prompt.wav",  # placeholder, not used when zero_shot_spk_id is set
                    zero_shot_spk_id=request.speaker_id,
                    stream=False,
                    speed=request.speed
                )
            else:
                # Mode 1: Traditional zero-shot, need audio
                if not request.prompt_audio:
                    raise HTTPException(status_code=400, detail="prompt_audio required for zero_shot mode when speaker_id is not provided")

                prompt_speech = decode_base64_audio(request.prompt_audio)
                temp_files.append(prompt_speech)
                model_output = model.inference_zero_shot(
                    tts_text=request.text,
                    prompt_text=request.prompt_text or "",
                    prompt_wav=prompt_speech,
                    stream=False,
                    speed=request.speed
                )

        elif mode == "cross_lingual":
            # Cross-lingual mode: synthesize in different language
            if not request.prompt_audio:
                raise HTTPException(status_code=400, detail="prompt_audio required for cross_lingual mode")

            prompt_speech = decode_base64_audio(request.prompt_audio)
            temp_files.append(prompt_speech)
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

            # Clean up instruct_text - remove duplicate <|endofprompt|> tags
            instruct_text = request.instruct_text
            while '<|endofprompt|><|endofprompt|>' in instruct_text:
                instruct_text = instruct_text.replace('<|endofprompt|><|endofprompt|>', '<|endofprompt|>')
            if instruct_text != request.instruct_text:
                logger.info(f"Cleaned up instruct_text (removed duplicate endofprompt tags)")

            # Debug: Check how text will be split
            logger.info("Text split preview:")
            split_texts = list(model.frontend.text_normalize(request.text, split=True, text_frontend=True))
            for i, split_text in enumerate(split_texts):
                logger.info(f"  Segment {i+1}: {split_text[:80]}{'...' if len(split_text) > 80 else ''}")
                logger.info(f"    Full text: [{split_text}]")
            logger.info(f"Total segments to process: {len(split_texts)}")

            # Use zero_shot_spk_id parameter for saved speakers
            # prompt_wav is required but not used when zero_shot_spk_id is provided
            # Use a default reference audio as placeholder
            default_prompt_wav = "CosyVoice/asset/zero_shot_prompt.wav"

            model_output = model.inference_instruct2(
                tts_text=request.text,
                instruct_text=instruct_text,
                prompt_wav=default_prompt_wav,
                zero_shot_spk_id=request.speaker_id,
                stream=False,
                speed=request.speed
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")

        # Collect audio chunks
        audio_chunks = []
        chunk_count = 0
        for output in model_output:
            audio = output['tts_speech']
            chunk_len = audio.shape[1] / model.sample_rate
            logger.info(f"  Collecting audio chunk {chunk_count + 1}: {chunk_len:.2f}s")
            audio_chunks.append(audio.numpy().flatten())
            chunk_count += 1
        logger.info(f"Total audio chunks collected: {chunk_count}")

        # Combine and convert to int16
        combined_audio = np.concatenate(audio_chunks)
        audio_int16 = (combined_audio * (2 ** 15)).astype(np.int16)
        audio_tensor = torch.from_numpy(audio_int16).unsqueeze(0)  # Add channel dimension

        # Write to temporary WAV file
        fd, temp_wav_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        torchaudio.save(temp_wav_path, audio_tensor, model.sample_rate, backend='soundfile')

        # Read back as bytes for proper WAV format
        with open(temp_wav_path, 'rb') as f:
            audio_bytes = f.read()

        # Clean up temp file
        os.remove(temp_wav_path)

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
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_file}: {e}")


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
