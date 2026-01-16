# VoiceHub Backend API 文档

基于 CosyVoice3 的语音合成 API，支持多种推理模式。

## 基础信息

- **Base URL**: `http://localhost:9880`
- **Content-Type**: `application/json`
- **Audio Encoding**: Base64

---

## API 端点

### 1. 健康检查

检查服务状态和模型信息。

**请求**
```http
GET /health
```

**响应**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "3",
  "speakers_count": 0,
  "available_modes": ["zero_shot", "cross_lingual", "instruct2"]
}
```

---

### 2. 列出说话人

获取所有可用的说话人 ID 列表。

**请求**
```http
GET /speakers
```

**响应**
```json
{
  "speakers": ["speaker_1", "speaker_2"],
  "count": 2,
  "mode": "sft"
}
```

---

### 3. 创建自定义说话人

使用参考音频创建自定义说话人，用于 Instruct2 模式。

**请求**
```http
POST /speakers
Content-Type: application/json

{
  "speaker_id": "my_voice",
  "prompt_text": "参考音频对应的文本内容",
  "prompt_audio": "base64编码的wav音频"
}
```

**参数说明**

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| speaker_id | string | ✅ | 唯一的说话人标识符 |
| prompt_text | string | ✅ | 参考音频对应的文本 |
| prompt_audio | string | ✅ | Base64 编码的 WAV 音频 |

**响应**
```json
{
  "success": true,
  "speaker_id": "my_voice",
  "message": "Speaker 'my_voice' created successfully"
}
```

**示例 (Python)**
```python
import base64
import requests

# 读取音频文件
with open("reference.wav", "rb") as f:
    audio_data = base64.b64encode(f.read()).decode("utf-8")

# 创建说话人
response = requests.post("http://localhost:9880/speakers", json={
    "speaker_id": "my_voice",
    "prompt_text": "希望你以后能够做的比我还好呦。",
    "prompt_audio": audio_data
})
```

---

### 4. 删除说话人

删除指定的自定义说话人。

**请求**
```http
DELETE /speakers/{speaker_id}
```

**响应**
```json
{
  "success": true,
  "message": "Speaker 'my_voice' deleted successfully"
}
```

---

### 5. 文字转语音 (TTS)

根据指定模式生成语音。

**请求**
```http
POST /tts
Content-Type: application/json

{
  "mode": "zero_shot",
  "text": "要合成的文本",
  "speed": 1.0,
  ...
}
```

**通用参数**

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| mode | string | ✅ | - | 推理模式: zero_shot, cross_lingual, instruct2 |
| text | string | ✅ | - | 要合成的文本 (1-5000字符) |
| speed | float | ❌ | 1.0 | 语速 (0.5-2.0) |
| seed | int | ❌ | - | 随机种子，用于复现结果 |

**响应**
```json
{
  "success": true,
  "audio_data": "base64编码的wav音频",
  "sample_rate": 24000,
  "duration": 5.2,
  "mode": "zero_shot"
}
```

---

## 推理模式详解

### Zero-Shot 模式

声音克隆模式，使用参考音频克隆音色。

**额外参数**

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| prompt_text | string | ✅ | 参考音频对应的文本 |
| prompt_audio | string | ✅ | Base64 编码的参考音频 |

**请求示例**
```json
{
  "mode": "zero_shot",
  "text": "你好，这是一个测试。",
  "prompt_text": "希望你以后能够做的比我还好呦。",
  "prompt_audio": "base64...",
  "speed": 1.0
}
```

**适用场景**：克隆任何声音的音色

---

### Cross-Lingual 模式

跨语言模式，不需要参考音频的文本。

**额外参数**

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| prompt_audio | string | ✅ | Base64 编码的参考音频 |

**请求示例**
```json
{
  "mode": "cross_lingual",
  "text": "Hello, this is a test.",
  "prompt_audio": "base64..."
}
```

**适用场景**：中文音频 → 英文输出（或任何语言组合）

---

### Instruct2 模式

情感/语调控制模式，需要先创建 speaker。

**额外参数**

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| speaker_id | string | ✅ | 已创建的说话人 ID |
| instruct_text | string | ✅ | 指令文本 |

**请求示例**
```json
{
  "mode": "instruct2",
  "text": "今天天气真好",
  "speaker_id": "my_voice",
  "instruct_text": "用欢快活泼的语气说这句话<|endofprompt|>"
}
```

**支持的指令**

| 指令类型 | 示例 |
|---------|------|
| 方言 | 用四川话说这句话 |
| 语速 | 请用尽可能快地语速说这句话 |
| 情感 | 用悲伤/欢快的语气说这句话 |

---

## 完整使用示例

### Python

```python
import base64
import requests

BASE_URL = "http://localhost:9880"

def read_audio(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# 1. 创建说话人
speaker = requests.post(f"{BASE_URL}/speakers", json={
    "speaker_id": "my_voice",
    "prompt_text": "希望你以后能够做的比我还好呦。",
    "prompt_audio": read_audio("reference.wav")
}).json()

# 2. Zero-Shot 模式
result = requests.post(f"{BASE_URL}/tts", json={
    "mode": "zero_shot",
    "text": "你好，这是测试。",
    "prompt_text": "希望你以后能够做的比我还好呦。",
    "prompt_audio": read_audio("reference.wav")
}).json()

# 3. 保存音频
with open("output.wav", "wb") as f:
    f.write(base64.b64decode(result["audio_data"]))
```

### cURL

```bash
# 健康检查
curl http://localhost:9880/health

# 列出说话人
curl http://localhost:9880/speakers

# Zero-Shot 模式
curl -X POST http://localhost:9880/tts \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "zero_shot",
    "text": "你好，这是测试。",
    "prompt_text": "希望你以后能够做的比我还好呦。",
    "prompt_audio": "<base64_audio>"
  }'
```

### JavaScript

```javascript
const BASE_URL = "http://localhost:9880";

// Zero-Shot 模式
async function textToSpeech(text, promptText, audioBase64) {
  const response = await fetch(`${BASE_URL}/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      mode: 'zero_shot',
      text: text,
      prompt_text: promptText,
      prompt_audio: audioBase64
    })
  });
  return await response.json();
}

// 使用示例
const result = await textToSpeech(
  "你好，这是测试。",
  "希望你以后能够做的比我还好呦。",
  audioBase64
);

// 保存音频
const audio = atob(result.audio_data);
// ... 处理音频数据
```

---

## 错误响应

错误格式：
```json
{
  "detail": "错误信息"
}
```

常见错误：
- `prompt_text required for zero_shot mode` - 缺少必需参数
- `speaker_id required for instruct2 mode` - 缺少说话人 ID
- `Speaker 'xxx' not found` - 说话人不存在
- `TTS generation failed: ...` - 生成失败

---

## 注意事项

1. **音频格式**：WAV 格式，采样率建议 22050Hz 或 24000Hz
2. **参考音频长度**：建议 3-10 秒，清晰无杂音
3. **文本长度**：建议单次不超过 500 字
4. **Speaker 持久化**：创建的 speaker 保存在 `spk2info.pt`，服务器重启后仍然存在
5. **API 文档**：访问 `http://localhost:9880/docs` 查看交互式 API 文档
