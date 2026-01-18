# VoiceHub Backend API 文档

基于 CosyVoice2 的语音合成 API，支持多种推理模式和系统监控。

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
  "model_version": "2",
  "model_name": "CosyVoice2-0.5B",
  "speakers_count": 5,
  "available_modes": ["zero_shot", "cross_lingual", "instruct2"],
  "uptime_seconds": 3600,
  "server_time": "2026-01-16T10:30:00Z"
}
```

**字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| status | string | 服务状态: healthy, unhealthy, loading |
| model_loaded | boolean | 模型是否已加载 |
| model_version | string | CosyVoice 版本号 |
| model_name | string | 模型名称 |
| speakers_count | int | 可用说话人数量 |
| available_modes | array | 支持的推理模式: sft, zero_shot, cross_lingual, instruct2 |
| uptime_seconds | int | 服务运行时长（秒） |
| server_time | string | 服务器时间（ISO 8601） |

---

### 2. 系统资源监控

获取系统资源使用情况，包括 CPU、内存、GPU、磁盘、网络。

**请求**
```http
GET /system/metrics
```

**响应**
```json
{
  "cpu": {
    "usage_percent": 25.5,
    "cores": 8
  },
  "memory": {
    "total_gb": 32.0,
    "used_gb": 12.5,
    "available_gb": 19.5,
    "usage_percent": 39.1
  },
  "gpu": {
    "available": true,
    "name": "NVIDIA GeForce RTX 3060",
    "driver_version": "536.99",
    "cuda_version": "12.2",
    "vram_total_mb": 12288,
    "vram_used_mb": 4300,
    "vram_free_mb": 7988,
    "gpu_utilization_percent": 45.2,
    "temperature_celsius": 52,
    "power_draw_watts": 145.5
  },
  "disk": {
    "total_gb": 512.0,
    "used_gb": 120.0,
    "available_gb": 392.0,
    "usage_percent": 23.4
  },
  "network": {
    "bytes_sent": 1024000,
    "bytes_recv": 2048000,
    "packets_sent": 5000,
    "packets_recv": 8000
  },
  "timestamp": "2026-01-16T10:30:00Z"
}
```

**字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| gpu.available | boolean | GPU 是否可用 |
| gpu.name | string | GPU 型号 |
| gpu.vram_total_mb | int | 显存总量 (MB) |
| gpu.vram_used_mb | int | 已用显存 (MB) |
| gpu.temperature_celsius | int | GPU 温度 (°C) |
| gpu.gpu_utilization_percent | float | GPU 利用率 (%) |

**错误响应**（GPU 不可用时）
```json
{
  "gpu": {
    "available": false,
    "error": "nvidia-smi not found"
  }
}
```

---

### 3. 列出说话人

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

### 4. 获取说话人详情

获取指定说话人的详细信息，包括 prompt_text。

**请求**
```http
GET /speakers/{speaker_id}
```

**响应**
```json
{
  "speaker_id": "my_voice",
  "prompt_text": "参考音频对应的文本内容",
  "is_builtin": false
}
```

**字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| speaker_id | string | 说话人 ID |
| prompt_text | string | 参考音频对应的文本（可能为 null） |
| is_builtin | boolean | 是否为内置说话人 |

---

### 5. 创建自定义说话人

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

---

### 6. 删除说话人

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

### 7. 文字转语音 (TTS)

根据指定模式生成语音。

**请求**
```http
POST /tts
Content-Type: application/json

{
  "mode": "zero_shot",
  "text": "要合成的文本",
  "speed": 1.0
}
```

**通用参数**

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| mode | string | ✅ | - | 推理模式: sft, zero_shot, cross_lingual, instruct2 |
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

### SFT 模式

预训练说话人模式，使用模型内置的说话人。

**额外参数**

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| speaker_id | string | ✅ | 内置说话人 ID |

**请求示例**
```json
{
  "mode": "sft",
  "text": "你好，这是一个测试。",
  "speaker_id": "中文女",
  "speed": 1.0
}
```

**注意**：仅当模型有预训练说话人时可用。

---

### Zero-Shot 模式

声音克隆模式，使用参考音频克隆音色。支持两种方式：

**方式一：传统模式（上传参考音频）**

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| prompt_text | string | ✅ | 参考音频对应的文本 |
| prompt_audio | string | ✅ | Base64 编码的参考音频 |

**方式二：使用已保存的说话人**

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| speaker_id | string | ✅ | 已创建的说话人 ID（无需重复上传音频） |

**请求示例（方式一）**
```json
{
  "mode": "zero_shot",
  "text": "你好，这是一个测试。",
  "prompt_text": "希望你以后能够做的比我还好呦。",
  "prompt_audio": "base64...",
  "speed": 1.0
}
```

**请求示例（方式二）**
```json
{
  "mode": "zero_shot",
  "text": "你好，这是一个测试。",
  "speaker_id": "my_voice",
  "speed": 1.0
}
```

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

---

## 前端集成示例

### JavaScript/TypeScript

```typescript
interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_version: string;
  model_name: string;
  speakers_count: number;
  available_modes: string[];
  uptime_seconds: number;
  server_time: string;
}

interface SystemMetrics {
  cpu: { usage_percent: number; cores: number };
  memory: { total_gb: number; used_gb: number; available_gb: number };
  gpu: {
    available: boolean;
    name?: string;
    vram_total_mb?: number;
    vram_used_mb?: number;
    temperature_celsius?: number;
    gpu_utilization_percent?: number;
  };
}

// 健康检查
async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch('http://localhost:9880/health');
  return await response.json();
}

// 系统监控（建议轮询间隔 2-5 秒）
async function getSystemMetrics(): Promise<SystemMetrics> {
  const response = await fetch('http://localhost:9880/system/metrics');
  return await response.json();
}

// 使用示例
const health = await checkHealth();
console.log(`Model: ${health.model_name}, Uptime: ${health.uptime_seconds}s`);

const metrics = await getSystemMetrics();
if (metrics.gpu.available) {
  console.log(`GPU: ${metrics.gpu.name}`);
  console.log(`VRAM: ${metrics.gpu.vram_used_mb} / ${metrics.gpu.vram_total_mb} MB`);
  console.log(`Temp: ${metrics.gpu.temperature_celsius}°C`);
}
```

---

## 注意事项

1. **音频格式**：WAV 格式，采样率建议 22050Hz 或 24000Hz
2. **参考音频长度**：建议 3-10 秒，清晰无杂音
3. **文本长度**：建议单次不超过 500 字
4. **Speaker 持久化**：创建的 speaker 保存在 `spk2info.pt`，服务器重启后仍然存在
5. **API 文档**：访问 `http://localhost:9880/docs` 查看交互式 API 文档
6. **GPU 监控**：需要 nvidia-smi 命令，无 GPU 时 `gpu.available` 为 false
