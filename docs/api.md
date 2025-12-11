# Whisper-WebUI REST API 接口文档
1
> 本文档基于后端代码（FastAPI）整理，供前端对接使用。
>
> 默认后端地址假设为：`http://localhost:8000`（请根据实际部署替换）。

---

## 一、总体说明

- **基础地址**：`http://<host>:<port>`，开发环境常见为 `http://localhost:8000`
- **接口风格**：RESTful
- **文档入口**：
  - Swagger UI：`/docs`（或根路径 `/` 自动重定向）
  - Redoc：`/redoc`
- **任务队列模式**：
  - 所有耗时操作（转写、VAD、BGM 分离）均采用异步任务队列。
  - 创建任务接口返回一个 `identifier`，前端需基于此调用任务查询接口轮询获取结果。

### 统一任务相关枚举

#### TaskStatus（任务状态）

```text
pending
queued
in_progress
completed
failed
cancelled
paused
retrying
```

#### TaskType（任务类型）

```text
transcription
vad
bgm_separation
```

#### ResultType（结果类型）

```text
json      # result 是 JSON 数据
filepath  # result 里保存的是文件路径信息（通常需要再调下载接口）
```

---

## 二、通用响应模型

### 1. QueueResponse（排队响应）

**出现位置**：所有“创建任务”的 POST 接口返回值。

```json
{
  "identifier": "string",  // 任务唯一 ID
  "status": "queued",      // 当前任务状态，初始通常为 queued
  "message": "Transcription task has queued"
}
```

字段说明：

- `identifier`：字符串，后续轮询 `/task/{identifier}` 的关键。
- `status`：任务当前状态（`TaskStatus` 枚举之一）。
- `message`：人类可读的提示信息。

### 2. TaskStatusResponse（任务状态查询）

**出现位置**：`GET /task/{identifier}`。

```json
{
  "identifier": "string",
  "status": "in_progress | completed | failed | ...",
  "task_type": "transcription | vad | bgm_separation",
  "result_type": "json | filepath",
  "result": {},
  "task_params": {},
  "error": "string or null",
  "duration": 12.34,
  "progress": 0.0
}
```

字段说明：

- `identifier`：任务 ID。
- `status`：当前任务状态（如 `queued`、`in_progress`、`completed`、`failed` 等）。
- `task_type`：任务类型（`transcription` / `vad` / `bgm_separation`）。
- `result_type`：结果类型（`json` / `filepath`）。
- `result`：任务结果内容，结构随任务类型而不同（详见后文各接口章节）。
- `task_params`：创建任务时的参数快照，方便调试或重放。
- `error`：失败时的错误消息，成功任务通常为 `null` 或空。
- `duration`：任务执行耗时（单位：秒，可能为空）。
- `progress`：进度，范围 `0.0 ~ 1.0`，部分任务会在处理中持续更新。

---

## 三、任务查询与结果下载

### 1. 查询任务状态

- **方法**：GET
- **路径**：`/task/{identifier}`
- **描述**：根据 `identifier` 查询指定任务当前状态、结果等信息。

#### 路径参数

- `identifier`：字符串，创建任务时返回的任务 ID。

#### 成功响应

- 状态码：`200 OK`
- Body：`TaskStatusResponse`（见上）

#### 可能的错误

- `404 Not Found`：任务不存在（identifier 错误或任务已被删除）。

#### 前端轮询示例（伪代码）

```js
async function pollTask(identifier, intervalMs = 2000, maxTry = 60) {
  for (let i = 0; i < maxTry; i++) {
    const res = await fetch(`/task/${identifier}`);
    if (!res.ok) throw new Error("Task not found");

    const data = await res.json();

    if (data.status === "completed") return data;      // 成功
    if (data.status === "failed") throw new Error(data.error || "Task failed");

    await new Promise(r => setTimeout(r, intervalMs));  // 间隔轮询
  }
  throw new Error("Task polling timeout");
}
```

### 2. 下载 BGM 分离结果文件

> 仅对 `task_type = bgm_separation` 的任务有效。

- **方法**：GET
- **路径**：`/task/file/{identifier}`
- **描述**：基于 BGM 分离任务的 `identifier` 下载压缩后的分离音频文件（ZIP）。

#### 路径参数

- `identifier`：BGM 分离任务的 ID。

#### 成功响应

- 状态码：`200 OK`
- Headers：
  - `Content-Type: application/zip`
- Body：ZIP 文件（二进制流），通常包含：
  - 伴奏文件（instrumental）
  - 人声文件（vocals）

#### 失败情况

- `404 Not Found`：
  - identifier 不存在；
  - 或任务类型不是 `bgm_separation`（接口只支持 BGM 分离结果的文件下载）。

#### 前端使用建议

- 建议在 `status === "completed"` 且 `task_type === "bgm_separation"` 时调用。
- 前端以 `blob` 形式接收并触发浏览器下载，例如：

```js
async function downloadBgmResult(identifier) {
  const res = await fetch(`/task/file/${identifier}`);
  if (!res.ok) throw new Error("download error");
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${identifier}_bgm_separation.zip`;
  a.click();
  URL.revokeObjectURL(url);
}
```

---

## 四、语音转写接口（Transcription）

### 1. 创建转写任务

- **方法**：POST
- **路径**：`/transcription/`
- **标签**：`Transcription`
- **描述**：上传音频/视频文件，创建异步语音转写任务。
- **请求 Content-Type**：`multipart/form-data`

#### 请求体字段

1）文件字段

- 字段名：`file`
- 类型：文件（二进制）
- 说明：音频或视频文件，后端自动解码。

2）查询参数（WhisperParams，转写参数）

> 所有参数均为 Query 参数，不传时使用默认值。前端可按需暴露常用参数，其余使用默认即可。

核心字段（节选）：

- `model_size`：`string`，默认 `"large-v2"`，Whisper 模型大小。
- `lang`：`string?`，语言代码，如 `"en"`、`"zh"`，为空或特殊值时表示自动检测。
- `is_translate`：`boolean`，默认 `false`，是否将语音直接翻译为英文。
- `beam_size`：`number`，默认 `5`，解码 beam 大小。
- `temperature`：`number`，默认 `0.0`，采样温度。
- `compute_type`：`string`，默认 `"float16"`，推理精度，比如 `"float16"` / `"int8"` / `"int16"`。
- `word_timestamps`：`boolean`，默认 `false`，是否返回逐词时间戳。
- `batch_size`：`number`，默认 `24`，批大小。

更多高阶参数（如 `log_prob_threshold`、`length_penalty` 等）均在 `modules/whisper/data_classes.py` 的 `WhisperParams` 中定义，如前端需要可继续补充开放。

3）查询参数（VadParams，VAD 相关）

> 这些参数控制在**转写流水线内部**是否及如何应用 VAD（裁剪静音）。

- `vad_filter`：`boolean`，默认 `false`，是否启用 VAD 过滤静音。
- `threshold`：`number`，范围 `0~1`，默认 `0.5`，VAD 概率阈值。
- `min_speech_duration_ms`：`number`，默认 `250`，小于此长度的语音片段将被丢弃（毫秒）。
- `max_speech_duration_s`：`number`，默认 `inf`，单段语音最大长度（秒）。
- `min_silence_duration_ms`：`number`，默认 `2000`，语音片段之间最短静音间隔（毫秒）。
- `speech_pad_ms`：`number`，默认 `400`，在语音片段两侧填充的时长（毫秒）。

4）查询参数（BGMSeparationParams，转写中的 BGM 预处理）

> 注意：这是**转写流水线内部**的 BGM 分离开关，与单独的 `/bgm-separation` 接口不同。

- `is_separate_bgm`：`boolean`，默认 `false`，是否在转写前先尝试 BGM 分离。
- `uvr_model_size`：`string`，默认 `"UVR-MDX-NET-Inst_HQ_4"`，UVR 模型名。
- `uvr_device`：`string`，默认 `"cuda"`，设备（如 `"cpu"` / `"cuda"` / `"xpu"`）。
- `segment_size`：`number`，默认 `256`，分段大小。
- `save_file`：`boolean`，默认 `false`，是否保存分离后的文件。
- `enable_offload`：`boolean`，默认 `true`，任务结束后是否卸载模型。

5）查询参数（DiarizationParams，说话人分离）

- `is_diarize`：`boolean`，默认 `false`，是否进行说话人分离。
- `diarization_device`：`string`，默认 `"cuda"`，设备。
- `hf_token`：`string?`，下载 diarization 模型时所需的 HuggingFace token（首次使用时需要）。
- `enable_offload`：`boolean`，默认 `true`，结束后是否卸载模型。

#### 成功响应

- 状态码：`201 Created`
- Body：`QueueResponse`

```json
{
  "identifier": "b1c0c6c4-...-...",
  "status": "queued",
  "message": "Transcription task has queued"
}
```

#### 任务完成后的结果结构

当通过 `GET /task/{identifier}` 查询到：

- `task_type = "transcription"`
- `status = "completed"`

时：

- `result` 字段为 `Segment` 列表。

`Segment`（简化）示例：

```json
{
  "id": 0,
  "seek": 0,
  "text": "Hello world",
  "start": 0.0,
  "end": 1.23,
  "tokens": [1, 2, 3],
  "temperature": 0.0,
  "avg_logprob": -0.1,
  "compression_ratio": 1.2,
  "no_speech_prob": 0.01,
  "words": [
    {
      "start": 0.0,
      "end": 0.5,
      "word": "Hello",
      "probability": 0.95
    }
  ]
}
```

前端典型用法：

- 将 `result` 按 `start/end` 渲染为时间轴字幕；
- 或导出为 SRT/VTT 等字幕文件格式。

---

## 五、VAD 接口（Voice Activity Detection）

### 1. 创建 VAD 任务

- **方法**：POST
- **路径**：`/vad/`
- **标签**：`Voice Activity Detection`
- **描述**：对上传的音频/视频文件执行语音活动检测，输出语音区间时间线。
- **请求 Content-Type**：`multipart/form-data`

#### 请求体字段

1）文件字段

- 字段名：`file`
- 类型：文件（二进制）
- 说明：音频或视频文件。

2）查询参数（VadParams，与上文一致）

- `vad_filter`：`boolean`，默认 `false`。
- `threshold`：`number`，默认 `0.5`。
- `min_speech_duration_ms`：`number`，默认 `250`。
- `max_speech_duration_s`：`number`，默认 `inf`。
- `min_silence_duration_ms`：`number`，默认 `2000`。
- `speech_pad_ms`：`number`，默认 `400`。

#### 成功响应

- 状态码：`201 Created`
- Body：`QueueResponse`

```json
{
  "identifier": "xxx",
  "status": "queued",
  "message": "VAD task has queued"
}
```

#### 任务完成后的结果结构

当通过 `GET /task/{identifier}` 查询到：

- `task_type = "vad"`
- `status = "completed"`

时：

- `result` 为 `speech_chunks` 列表，每个元素是一个字典：

```json
[
  { "start": 1600, "end": 6400 },
  { "start": 8000, "end": 12000 }
]
```

说明：

- 采样率固定为 `16000 Hz`；
- `start` / `end` 为采样点索引（sample index）；
- 对应时间（秒） = `index / 16000`。

前端可用这些时间段：

- 在波形图/进度条上高亮出“有声”片段；
- 联动播放器 `currentTime` 做可视化提示。

---

## 六、BGM 分离接口（Background Music Separation）

### 1. 创建 BGM 分离任务

- **方法**：POST
- **路径**：`/bgm-separation/`
- **标签**：`BGM Separation`
- **描述**：将上传的音频/视频中的人声与伴奏分离。
- **请求 Content-Type**：`multipart/form-data`

#### 请求体字段

1）文件字段

- 字段名：`file`
- 类型：文件（二进制）
- 说明：音频或视频文件。

2）查询参数（BGMSeparationParams，与转写内部使用的同一套）

> 为了真正触发分离操作，通常需要将 `is_separate_bgm` 设为 `true`。

- `is_separate_bgm`：`boolean`，默认 `false`，是否启用 BGM 分离。
- `uvr_model_size`：`string`，默认 `"UVR-MDX-NET-Inst_HQ_4"`，UVR 模型名。
- `uvr_device`：`string`，默认 `"cuda"`，可选 `"cpu"` / `"cuda"` / `"xpu"`。
- `segment_size`：`number`，默认 `256`，分段大小。
- `save_file`：`boolean`，默认 `false`，是否持久化分离后的音频文件。
- `enable_offload`：`boolean`，默认 `true`，任务结束后是否卸载模型。

#### 成功响应

- 状态码：`201 Created`
- Body：`QueueResponse`

```json
{
  "identifier": "xxx",
  "status": "queued",
  "message": "BGM Separation task has queued"
}
```

### 2. 任务完成后的结果结构

当通过 `GET /task/{identifier}` 查询到：

- `task_type = "bgm_separation"`
- `status = "completed"`

时：

- `result_type = "filepath"`
- `result` 字段结构：

```json
{
  "instrumental_hash": "string",
  "vocal_hash": "string"
}
```

> 真实文件路径由后端在缓存目录中通过哈希查找，不直接暴露在此 JSON 中。

如需下载实际音频文件，请调用前文的：

- `GET /task/file/{identifier}`

返回值为 ZIP 文件，通常包含：

- 伴奏音轨（instrumental）
- 人声音轨（vocals）

前端典型流程：

1. `POST /bgm-separation/` 创建任务，拿到 `identifier`；
2. 轮询 `GET /task/{identifier}`，直到 `status === "completed"`；
3. 调用 `GET /task/file/{identifier}` 下载 zip；
4. 在浏览器中触发下载或在本地解压进一步处理。

---

## 七、前端对接建议

1. **统一封装创建任务**

   - 对 `/transcription/`、`/vad/`、`/bgm-separation/` 封装统一的“创建任务”方法：
     - 使用 `multipart/form-data` 上传文件字段 `file`；
     - 其它参数以 Query 方式附加在 URL 上；
     - 不传的参数使用后端默认值。

2. **统一封装任务轮询**

   - 抽象出公共 `pollTask(identifier)` 工具，内部调用 `GET /task/{identifier}` 轮询。
   - 支持配置轮询间隔与最大次数，便于 UI 做 loading 与超时提示。

3. **按任务类型解析结果**

   - `task_type === "transcription"`：
     - `result` 为字幕片段（Segment）数组，可直接按时间渲染字幕或做文本导出。
   - `task_type === "vad"`：
     - `result` 为语音区间数组，需注意根据采样率换算为秒。
   - `task_type === "bgm_separation"`：
     - `result` 仅包含哈希信息，需配合 `GET /task/file/{identifier}` 下载 ZIP 再处理。

4. **错误与状态处理**

   - 注意处理：
     - `status === "failed"` 时，优先展示 `error` 字段内容；
     - `404` 情况（如 identifier 过期或被删除）；
     - 长时间 `queued / in_progress` 时给用户相应的 UI 提示。

---

如需扩展文档（例如增加鉴权、分页、前端具体请求示例等），可以在本文件基础上继续补充。