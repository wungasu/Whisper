// 使用环境变量 VITE_API_BASE 作为后端基础地址，默认 http://localhost:8000

let API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export function setApiBase(base) {
  if (base) {
    API_BASE = base.replace(/\/$/, '')
  }
}

function buildQuery(params = {}) {
  const searchParams = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') return
    searchParams.append(key, String(value))
  })
  const queryString = searchParams.toString()
  return queryString ? `?${queryString}` : ''
}

async function requestJson(path, { method = 'GET', body, headers } = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    body,
    headers,
  })

  if (!res.ok) {
    let message = `HTTP ${res.status}`
    try {
      const data = await res.json()
      if (data && data.detail) {
        message += `: ${data.detail}`
      }
    } catch (_) {
      // ignore
    }
    throw new Error(message)
  }

  return res.json()
}

// 创建转写任务
export async function createTranscriptionTask({ file, whisperParams = {}, vadParams = {}, bgmParams = {}, diarizationParams = {} }) {
  if (!file) {
    throw new Error('请先选择要上传的音频/视频文件')
  }

  const allParams = {
    // WhisperParams
    ...whisperParams,
    // VadParams
    ...vadParams,
    // BGMSeparationParams（流水线内部）
    ...bgmParams,
    // DiarizationParams
    ...diarizationParams,
  }

  const query = buildQuery(allParams)
  const formData = new FormData()
  formData.append('file', file)

  return requestJson(`/transcription/${query}`, {
    method: 'POST',
    body: formData,
  })
}

// 创建 VAD 任务
export async function createVadTask({ file, vadParams = {} }) {
  if (!file) {
    throw new Error('请先选择要上传的音频/视频文件')
  }

  const query = buildQuery(vadParams)
  const formData = new FormData()
  formData.append('file', file)

  return requestJson(`/vad/${query}`, {
    method: 'POST',
    body: formData,
  })
}

// 创建 BGM 分离任务
export async function createBgmSeparationTask({ file, bgmParams = {} }) {
  if (!file) {
    throw new Error('请先选择要上传的音频/视频文件')
  }

  const query = buildQuery(bgmParams)
  const formData = new FormData()
  formData.append('file', file)

  return requestJson(`/bgm-separation/${query}`, {
    method: 'POST',
    body: formData,
  })
}

// 轮询任务状态，直到完成 / 失败 / 超时
export async function pollTask(identifier, {
  intervalMs = 2000,
  maxAttempts = 60,
  onUpdate,
} = {}) {
  let lastStatus = null
  for (let i = 0; i < maxAttempts; i++) {
    const status = await requestJson(`/task/${identifier}`)
    lastStatus = status

    if (typeof onUpdate === 'function') {
      try {
        onUpdate(status)
      } catch (_) {
        // 忽略 UI 回调中的异常
      }
    }

    if (status.status === 'completed') {
      return status
    }

    if (status.status === 'failed') {
      const msg = status.error || '任务失败'
      throw new Error(msg)
    }

    await new Promise((resolve) => setTimeout(resolve, intervalMs))
  }

  throw new Error('轮询任务超时')
}

// 下载 BGM 分离结果 ZIP
export async function downloadBgmZip(identifier) {
  const res = await fetch(`${API_BASE}/task/file/${identifier}`)
  if (!res.ok) {
    throw new Error(`下载失败：HTTP ${res.status}`)
  }
  return res.blob()
}
