# VoiceHub Backend

Minimal TTS backend for [VoiceHub](https://github.com/youyouhe/VoiceHub) frontend using CosyVoice.

## Features

- 🚀 Simple, lightweight FastAPI server
- 🎙️ Text-to-Speech using CosyVoice models
- 🌍 Multi-language support (Chinese, English, Japanese, Korean, etc.)
- 📊 OpenAPI documentation at `/docs`
- 🔊 Base64-encoded audio output
- ⚡ Configurable speaker selection and speed

## Quick Start

### 1. Clone CosyVoice (as submodule)

```bash
git submodule add https://github.com/FunAudioLLM/CosyVoice.git CosyVoice
cd CosyVoice
git submodule update --init --recursive
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Model (auto-downloads on first run)

```bash
# Or manually download from ModelScope
python -c "from modelscope import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B')"
```

### 4. Run Server

```bash
python server.py --port 9880
```

## API Endpoints

### `GET /health`
Health check and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "3",
  "speakers_count": 20
}
```

### `GET /speakers`
List available pretrained speakers.

**Response:**
```json
{
  "speakers": ["中文女", "中文男", "英文女", ...],
  "count": 20
}
```

### `POST /tts`
Generate speech from text.

**Request:**
```json
{
  "text": "你好，世界",
  "speaker_id": "中文女",
  "speed": 1.0,
  "seed": 42
}
```

**Response:**
```json
{
  "success": true,
  "audio_data": "<base64>",
  "sample_rate": 24000,
  "duration": 2.5
}
```

## Configuration

Environment variables or command-line arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `0.0.0.0` | Server host |
| `--port` | `9880` | Server port |
| `--model-dir` | `CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B` | Model directory |

## Usage with VoiceHub Frontend

1. Start this backend server
2. In VoiceHub frontend settings, set backend URL to: `http://localhost:9880`
3. Select a model and speaker
4. Generate speech!

## Docker Deployment

```bash
docker build -t voicehub-backend .
docker run -d --gpus all -p 9880:9880 voicehub-backend
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 2GB+ disk space for model

## License

Apache 2.0

## Links

- [VoiceHub Frontend](https://github.com/youyouhe/VoiceHub)
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
