#!/usr/bin/env python3
"""Test script for zero-shot TTS mode."""

import base64
import requests
import json

def read_audio_to_base64(filepath):
    """Read audio file and encode to base64."""
    with open(filepath, 'rb') as f:
        audio_data = f.read()
    return base64.b64encode(audio_data).decode('utf-8')

def test_zero_shot():
    """Test zero-shot TTS mode."""
    url = "http://localhost:9880/tts"

    # Read reference audio
    prompt_audio = read_audio_to_base64("CosyVoice/asset/zero_shot_prompt.wav")

    payload = {
        "mode": "zero_shot",
        "text": "你好，这是一个语音合成测试。",
        "prompt_text": "希望你以后能够做的比我还好呦。",
        "prompt_audio": prompt_audio,
        "speed": 1.0
    }

    print("Testing zero-shot mode...")
    print(f"Text: {payload['text']}")
    print(f"Prompt text: {payload['prompt_text']}")
    print(f"Reference audio: CosyVoice/asset/zero_shot_prompt.wav")

    response = requests.post(url, json=payload, timeout=60)

    print(f"\nStatus: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Sample rate: {result['sample_rate']} Hz")
        print(f"Duration: {result['duration']:.2f} seconds")
        print(f"Mode: {result['mode']}")
        print(f"Audio data size: {len(result['audio_data'])} bytes (base64)")

        # Save audio to file
        audio_bytes = base64.b64decode(result['audio_data'])
        output_file = "test_output.wav"
        with open(output_file, 'wb') as f:
            f.write(audio_bytes)
        print(f"\nAudio saved to: {output_file}")

        # Calculate raw audio size
        raw_size = len(audio_bytes)
        print(f"Raw audio size: {raw_size} bytes ({raw_size / 1024:.2f} KB)")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_zero_shot()
