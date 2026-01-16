#!/usr/bin/env python3
"""Test script for zero-shot mode with instruct commands."""

import base64
import requests

def read_audio_to_base64(filepath):
    """Read audio file and encode to base64."""
    with open(filepath, 'rb') as f:
        audio_data = f.read()
    return base64.b64encode(audio_data).decode('utf-8')

def test_zero_shot_instruct():
    """Test zero-shot mode with instruct commands."""
    url = "http://localhost:9880/tts"

    # Read reference audio
    prompt_audio = read_audio_to_base64("CosyVoice/asset/zero_shot_prompt.wav")

    # Test cases with different instruct commands
    test_cases = [
        {
            "name": "普通模式",
            "text": "今天天气真好，阳光明媚，微风不燥，正是出去踏青的好时节。公园里的花开得正艳，鸟儿在枝头欢快地歌唱，孩子们在草地上奔跑嬉戏，一派生机勃勃的景象。",
            "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        },
        {
            "name": "四川话",
            "text": "今天天气真好，阳光明媚，微风不燥，正是出去踏青的好时节。公园里的花开得正艳，鸟儿在枝头欢快地歌唱，孩子们在草地上奔跑嬉戏，一派生机勃勃的景象。",
            "prompt_text": "You are a helpful assistant. 用四川话说这句话<|endofprompt|>",
        },
        {
            "name": "快速语速",
            "text": "今天天气真好，阳光明媚，微风不燥，正是出去踏青的好时节。公园里的花开得正艳，鸟儿在枝头欢快地歌唱，孩子们在草地上奔跑嬉戏，一派生机勃勃的景象。",
            "prompt_text": "You are a helpful assistant. 请用尽可能快地语速说这句话。<|endofprompt|>",
        },
        {
            "name": "悲伤语气",
            "text": "今天天气真好，阳光明媚，微风不燥，正是出去踏青的好时节。公园里的花开得正艳，鸟儿在枝头欢快地歌唱，孩子们在草地上奔跑嬉戏，一派生机勃勃的景象。",
            "prompt_text": "You are a helpful assistant. 用悲伤的语气说这句话<|endofprompt|>",
        },
        {
            "name": "欢快语气",
            "text": "今天天气真好，阳光明媚，微风不燥，正是出去踏青的好时节。公园里的花开得正艳，鸟儿在枝头欢快地歌唱，孩子们在草地上奔跑嬉戏，一派生机勃勃的景象。",
            "prompt_text": "You are a helpful assistant. 用欢快活泼的语气说这句话<|endofprompt|>",
        },
    ]

    for i, case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"测试 {i+1}: {case['name']}")
        print(f"{'='*60}")
        print(f"Text: {case['text']}")
        print(f"Prompt text: {case['prompt_text']}")

        payload = {
            "mode": "zero_shot",
            "text": case['text'],
            "prompt_text": case['prompt_text'],
            "prompt_audio": prompt_audio,
            "speed": 1.0
        }

        response = requests.post(url, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success - Duration: {result['duration']:.2f}s")

            # Save audio to file
            audio_bytes = base64.b64decode(result['audio_data'])
            output_file = f"test_instruct_{i+1}_{case['name']}.wav"
            with open(output_file, 'wb') as f:
                f.write(audio_bytes)
            print(f"   Saved to: {output_file}")
        else:
            print(f"❌ Error: {response.text}")

if __name__ == "__main__":
    test_zero_shot_instruct()
