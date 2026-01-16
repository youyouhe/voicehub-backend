#!/usr/bin/env python3
"""Test script for Instruct2 mode with custom speaker."""

import base64
import requests
import time


def read_audio_to_base64(filepath):
    """Read audio file and encode to base64."""
    with open(filepath, 'rb') as f:
        audio_data = f.read()
    return base64.b64encode(audio_data).decode('utf-8')


def create_speaker(speaker_id, prompt_text, prompt_audio):
    """Create a custom speaker using zero-shot."""
    url = "http://localhost:9880/speakers"
    payload = {
        "speaker_id": speaker_id,
        "prompt_text": prompt_text,
        "prompt_audio": prompt_audio
    }
    response = requests.post(url, json=payload, timeout=60)
    return response.json()


def list_speakers():
    """List all available speakers."""
    url = "http://localhost:9880/speakers"
    response = requests.get(url, timeout=10)
    return response.json()


def delete_speaker(speaker_id):
    """Delete a speaker by ID."""
    url = f"http://localhost:9880/speakers/{speaker_id}"
    response = requests.delete(url, timeout=10)
    return response.json()


def test_tts(mode, text, **kwargs):
    """Test TTS generation."""
    url = "http://localhost:9880/tts"
    payload = {
        "mode": mode,
        "text": text,
        "speed": 1.0,
        **kwargs
    }
    response = requests.post(url, json=payload, timeout=60)
    return response.json()


def test_instruct2_with_custom_speaker():
    """Test Instruct2 mode with custom speaker."""
    print("=" * 60)
    print("Instruct2 Mode Test with Custom Speaker")
    print("=" * 60)

    # Read reference audio
    prompt_audio = read_audio_to_base64("CosyVoice/asset/zero_shot_prompt.wav")
    prompt_text = "希望你以后能够做的比我还好呦。"

    # Define speaker ID
    speaker_id = "my_custom_speaker"

    # Step 1: List existing speakers
    print("\n[Step 1] Listing existing speakers...")
    speakers = list_speakers()
    print(f"  Existing speakers: {speakers['speakers']}")
    print(f"  Count: {speakers['count']}")

    # Step 2: Create custom speaker
    print(f"\n[Step 2] Creating custom speaker '{speaker_id}'...")
    print(f"  Prompt text: {prompt_text}")

    # Delete speaker if exists
    if speaker_id in speakers['speakers']:
        print(f"  Speaker already exists, deleting first...")
        delete_speaker(speaker_id)

    result = create_speaker(speaker_id, prompt_text, prompt_audio)
    if result.get('success'):
        print(f"  ✅ Speaker created: {speaker_id}")
    else:
        print(f"  ❌ Failed to create speaker: {result}")
        return

    # Step 3: List speakers again
    print("\n[Step 3] Listing speakers after creation...")
    speakers = list_speakers()
    print(f"  Speakers: {speakers['speakers']}")
    print(f"  Count: {speakers['count']}")

    # Step 4: Test Instruct2 mode with different instructions
    text = "今天天气真好，阳光明媚，微风不燥，正是出去踏青的好时节。"
    test_cases = [
        {
            "name": "普通模式（无指令）",
            "instruct_text": None,
        },
        {
            "name": "快速语速",
            "instruct_text": "请用尽可能快地语速说这句话。<|endofprompt|>",
        },
        {
            "name": "悲伤语气",
            "instruct_text": "用悲伤的语气说这句话<|endofprompt|>",
        },
        {
            "name": "欢快语气",
            "instruct_text": "用欢快活泼的语气说这句话<|endofprompt|>",
        },
    ]

    print(f"\n[Step 4] Testing Instruct2 mode...")
    print(f"  Text: {text}\n")

    for i, case in enumerate(test_cases, 1):
        print(f"  Test {i}: {case['name']}")
        if case['instruct_text']:
            print(f"    Instruction: {case['instruct_text']}")

        kwargs = {
            "speaker_id": speaker_id,
        }
        if case['instruct_text']:
            kwargs["instruct_text"] = case['instruct_text']

        result = test_tts("instruct2", text, **kwargs)

        if result.get('success'):
            print(f"    ✅ Success - Duration: {result['duration']:.2f}s")
            # Save audio
            audio_bytes = base64.b64decode(result['audio_data'])
            filename = f"test_instruct2_{i}_{case['name']}.wav"
            with open(filename, 'wb') as f:
                f.write(audio_bytes)
            print(f"    Saved to: {filename}")
        else:
            print(f"    ❌ Error: {result}")
        print()

    # Step 5: Cleanup - delete custom speaker
    print(f"\n[Step 5] Cleanup - deleting speaker '{speaker_id}'...")
    result = delete_speaker(speaker_id)
    print(f"  {result.get('message', 'Done')}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_instruct2_with_custom_speaker()
