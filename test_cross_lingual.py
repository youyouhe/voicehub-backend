#!/usr/bin/env python3
"""Test script for cross-lingual TTS mode."""

import base64
import requests
import json

def read_audio_to_base64(filepath):
    """Read audio file and encode to base64."""
    with open(filepath, 'rb') as f:
        audio_data = f.read()
    return base64.b64encode(audio_data).decode('utf-8')

def test_cross_lingual():
    """Test cross-lingual TTS mode."""
    url = "http://localhost:9880/tts"

    # Read reference audio (can be any language)
    prompt_audio = read_audio_to_base64("CosyVoice/asset/zero_shot_prompt.wav")

    payload = {
        "mode": "cross_lingual",
        "text": "Hello, this is a test of cross-lingual speech synthesis.",
        "prompt_audio": prompt_audio,
        "speed": 1.0
    }

    print("Testing cross-lingual mode...")
    print(f"Text: {payload['text']}")
    print(f"Reference audio: CosyVoice/asset/zero_shot_prompt.wav")
    print("Note: Reference audio is Chinese, target text is English")

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
        output_file = "test_output_cross_lingual.wav"
        with open(output_file, 'wb') as f:
            f.write(audio_bytes)
        print(f"\nAudio saved to: {output_file}")

        raw_size = len(audio_bytes)
        print(f"Raw audio size: {raw_size} bytes ({raw_size / 1024:.2f} KB)")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_cross_lingual()
