<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import {
  setApiBase,
  createTranscriptionTask,
  createVadTask,
  createBgmSeparationTask,
  pollTask,
  downloadBgmZip,
} from './api/backend'

const defaultApiBase = import.meta.env.VITE_API_BASE || 'http://localhost:8000'
const apiBaseInput = ref(defaultApiBase)
setApiBase(apiBaseInput.value)

// ------------------------- 顶部轮播图（博物馆风格） -------------------------
const carouselItems = [
  {
    src: '/archives/reading-room.jpg',
    title: '校史档案阅览室',
    desc: '珍藏学校发展历程中的重要文献与影像资料。',
  },
  {
    src: '/archives/exhibition-hall.jpg',
    title: '校史陈列展厅',
    desc: '以时间为轴，讲述校园的百年变迁与辉煌时刻。',
  },
  {
    src: '/archives/digital-archive.jpg',
    title: '数字化档案平台',
    desc: '借助智能语音技术，实现多媒体档案的智能检索与展示。',
  },
]

const carouselIndex = ref(0)
let carouselTimer = null

function nextSlide() {
  carouselIndex.value = (carouselIndex.value + 1) % carouselItems.length
}

function prevSlide() {
  carouselIndex.value = (carouselIndex.value - 1 + carouselItems.length) % carouselItems.length
}

function startCarousel() {
  stopCarousel()
  carouselTimer = setInterval(() => {
    nextSlide()
  }, 6000)
}

function stopCarousel() {
  if (carouselTimer) {
    clearInterval(carouselTimer)
    carouselTimer = null
  }
}

onMounted(() => {
  startCarousel()
})

onBeforeUnmount(() => {
  stopCarousel()
})

const activeTab = ref('transcription')

// 公共工具
function formatSeconds(seconds) {
  if (seconds == null) return '-'
  return `${seconds.toFixed(2)}s`
}

// ------------------------- 转写 -------------------------
const transFile = ref(null)
const transModelSize = ref('large-v2')
const transLang = ref('')
const transTranslate = ref(false)
const transUseVad = ref(false)
const transUseDiarization = ref(false)
const transUseBgm = ref(false)

const transTaskId = ref('')
const transStatus = ref('')
const transProgress = ref(0)
const transDuration = ref(null)
const transResult = ref([])
const transError = ref('')
const transLoading = ref(false)

const transText = computed(() => {
  if (!Array.isArray(transResult.value)) return ''
  return transResult.value.map((seg) => seg.text || '').join('\n')
})

function onTransFileChange(event) {
  const [file] = event.target.files || []
  transFile.value = file || null
}

async function handleTranscription() {
  if (!transFile.value) {
    transError.value = '请先选择要上传的文件'
    return
  }

  transError.value = ''
  transLoading.value = true
  transResult.value = []
  transTaskId.value = ''
  transStatus.value = 'queued'
  transProgress.value = 0
  transDuration.value = null

  try {
    const whisperParams = {
      model_size: transModelSize.value,
      lang: transLang.value || undefined,
      is_translate: transTranslate.value,
    }

    const vadParams = transUseVad.value
      ? {
          vad_filter: true,
        }
      : {}

    const bgmParams = transUseBgm.value
      ? {
          is_separate_bgm: true,
        }
      : {}

    const diarizationParams = transUseDiarization.value
      ? {
          is_diarize: true,
        }
      : {}

    const queue = await createTranscriptionTask({
      file: transFile.value,
      whisperParams,
      vadParams,
      bgmParams,
      diarizationParams,
    })

    transTaskId.value = queue.identifier
    transStatus.value = queue.status

    const finalStatus = await pollTask(queue.identifier, {
      onUpdate(status) {
        transStatus.value = status.status
        transProgress.value = status.progress ?? 0
      },
    })

    transResult.value = finalStatus.result || []
    transDuration.value = finalStatus.duration ?? null
  } catch (e) {
    transError.value = e.message || String(e)
  } finally {
    transLoading.value = false
  }
}

// ------------------------- VAD -------------------------
const vadFile = ref(null)
const vadThreshold = ref(0.5)
const vadMinSpeechMs = ref(250)
const vadTaskId = ref('')
const vadStatus = ref('')
const vadProgress = ref(0)
const vadResult = ref([])
const vadError = ref('')
const vadLoading = ref(false)

function onVadFileChange(event) {
  const [file] = event.target.files || []
  vadFile.value = file || null
}

async function handleVad() {
  if (!vadFile.value) {
    vadError.value = '请先选择要上传的文件'
    return
  }

  vadError.value = ''
  vadLoading.value = true
  vadResult.value = []
  vadTaskId.value = ''
  vadStatus.value = 'queued'
  vadProgress.value = 0

  try {
    const vadParams = {
      threshold: vadThreshold.value,
      min_speech_duration_ms: vadMinSpeechMs.value,
    }

    const queue = await createVadTask({
      file: vadFile.value,
      vadParams,
    })

    vadTaskId.value = queue.identifier
    vadStatus.value = queue.status

    const finalStatus = await pollTask(queue.identifier, {
      onUpdate(status) {
        vadStatus.value = status.status
        vadProgress.value = status.progress ?? 0
      },
    })

    vadResult.value = finalStatus.result || []
  } catch (e) {
    vadError.value = e.message || String(e)
  } finally {
    vadLoading.value = false
  }
}

// ------------------------- BGM 分离 -------------------------
const bgmFile = ref(null)
const bgmModelSize = ref('UVR-MDX-NET-Inst_HQ_4')
const bgmTaskId = ref('')
const bgmStatus = ref('')
const bgmProgress = ref(0)
const bgmResult = ref(null)
const bgmError = ref('')
const bgmLoading = ref(false)
const bgmDownloadUrl = ref('')

function onBgmFileChange(event) {
  const [file] = event.target.files || []
  bgmFile.value = file || null
}

async function handleBgmSeparation() {
  if (!bgmFile.value) {
    bgmError.value = '请先选择要上传的文件'
    return
  }

  bgmError.value = ''
  bgmLoading.value = true
  bgmResult.value = null
  bgmTaskId.value = ''
  bgmStatus.value = 'queued'
  bgmProgress.value = 0
  bgmDownloadUrl.value = ''

  try {
    const bgmParams = {
      is_separate_bgm: true,
      uvr_model_size: bgmModelSize.value,
    }

    const queue = await createBgmSeparationTask({
      file: bgmFile.value,
      bgmParams,
    })

    bgmTaskId.value = queue.identifier
    bgmStatus.value = queue.status

    const finalStatus = await pollTask(queue.identifier, {
      onUpdate(status) {
        bgmStatus.value = status.status
        bgmProgress.value = status.progress ?? 0
      },
    })

    bgmResult.value = finalStatus.result || null

    // 任务完成后自动获取 ZIP 下载链接
    const blob = await downloadBgmZip(queue.identifier)
    const url = URL.createObjectURL(blob)
    bgmDownloadUrl.value = url
  } catch (e) {
    bgmError.value = e.message || String(e)
  } finally {
    bgmLoading.value = false
  }
}

function applyApiBase() {
  setApiBase(apiBaseInput.value.trim() || defaultApiBase)
}
</script>

<template>
  <div class="app-root">
    <header class="app-header">
      <h1>Whisper-WebUI 前端示例</h1>
      <p class="subtitle">转写 / VAD / BGM 分离 一体化示例模板</p>

      <div class="api-base">
        <label>
          后端 API 地址：
          <input v-model="apiBaseInput" type="text" placeholder="http://localhost:8000" />
        </label>
        <button type="button" @click="applyApiBase">应用</button>
      </div>
    </header>

    <main class="content">
      <!-- 博物馆风格头图与轮播 -->
      <section class="hero">
        <div class="hero-text">
          <p class="hero-tag">校史·博物馆·档案馆</p>
          <h2>音视频档案智能整理平台</h2>
          <p class="hero-desc">
            面向高校博物馆与档案馆的数字化方案，支持音视频资料的自动转写、语音分段与背景音乐分离，
            为校史研究与展陈规划提供更便捷的工具支撑。
          </p>
        </div>
        <div class="hero-carousel" @mouseenter="stopCarousel" @mouseleave="startCarousel">
          <div class="carousel-window">
            <div
              v-for="(item, idx) in carouselItems"
              :key="item.title"
              class="carousel-slide"
              :class="{ active: idx === carouselIndex }"
            >
              <div class="carousel-image" :style="{ backgroundImage: `url(${item.src})` }"></div>
              <div class="carousel-caption">
                <h3>{{ item.title }}</h3>
                <p>{{ item.desc }}</p>
              </div>
            </div>
          </div>
          <button type="button" class="carousel-nav prev" @click="prevSlide">‹</button>
          <button type="button" class="carousel-nav next" @click="nextSlide">›</button>
          <div class="carousel-dots">
            <button
              v-for="(item, idx) in carouselItems"
              :key="item.title + idx"
              type="button"
              :class="['dot', { active: idx === carouselIndex }]"
              @click="carouselIndex = idx"
            />
          </div>
        </div>
      </section>

      <div class="tabs">
        <button
          type="button"
          class="tab-btn"
          :class="{ active: activeTab === 'transcription' }"
          @click="activeTab = 'transcription'"
        >
          语音转写
        </button>
        <button
          type="button"
          class="tab-btn"
          :class="{ active: activeTab === 'vad' }"
          @click="activeTab = 'vad'"
        >
          VAD 检测
        </button>
        <button
          type="button"
          class="tab-btn"
          :class="{ active: activeTab === 'bgm' }"
          @click="activeTab = 'bgm'"
        >
          BGM 分离
        </button>
      </div>

      <!-- 转写 -->
      <section v-if="activeTab === 'transcription'" class="panel">
        <h2>语音转写</h2>
        <div class="form-grid">
          <div class="form-item full-width">
            <label>
              音频 / 视频文件：
              <input type="file" accept="audio/*,video/*" @change="onTransFileChange" />
            </label>
          </div>

          <div class="form-item">
            <label>
              模型大小：
              <select v-model="transModelSize">
                <option value="tiny">tiny</option>
                <option value="base">base</option>
                <option value="small">small</option>
                <option value="medium">medium</option>
                <option value="large-v2">large-v2</option>
              </select>
            </label>
          </div>

          <div class="form-item">
            <label>
              语言（留空自动检测）：
              <input v-model="transLang" type="text" placeholder="例如: en, zh" />
            </label>
          </div>

          <div class="form-item">
            <label class="checkbox">
              <input v-model="transTranslate" type="checkbox" /> 翻译为英文
            </label>
          </div>

          <div class="form-item">
            <label class="checkbox">
              <input v-model="transUseVad" type="checkbox" /> 启用 VAD 过滤静音
            </label>
          </div>

          <div class="form-item">
            <label class="checkbox">
              <input v-model="transUseDiarization" type="checkbox" /> 启用说话人分离
            </label>
          </div>

          <div class="form-item">
            <label class="checkbox">
              <input v-model="transUseBgm" type="checkbox" /> 启用 BGM 预处理
            </label>
          </div>
        </div>

        <div class="actions">
          <button type="button" :disabled="transLoading" @click="handleTranscription">
            {{ transLoading ? '处理中...' : '开始转写' }}
          </button>
        </div>

        <div class="status" v-if="transTaskId || transError">
          <p v-if="transTaskId">任务 ID：{{ transTaskId }}</p>
          <p v-if="transStatus">状态：{{ transStatus }}，进度：{{ (transProgress * 100).toFixed(0) }}%</p>
          <p v-if="transDuration != null">耗时：{{ formatSeconds(transDuration) }}</p>
          <p v-if="transError" class="error">错误：{{ transError }}</p>
        </div>

        <div v-if="transText" class="result-box">
          <h3>转写结果</h3>
          <pre>{{ transText }}</pre>
        </div>
      </section>

      <!-- VAD -->
      <section v-else-if="activeTab === 'vad'" class="panel">
        <h2>VAD 检测</h2>
        <div class="form-grid">
          <div class="form-item full-width">
            <label>
              音频 / 视频文件：
              <input type="file" accept="audio/*,video/*" @change="onVadFileChange" />
            </label>
          </div>

          <div class="form-item">
            <label>
              阈值（0-1）：
              <input v-model.number="vadThreshold" type="number" min="0" max="1" step="0.01" />
            </label>
          </div>

          <div class="form-item">
            <label>
              最短语音时长（毫秒）：
              <input v-model.number="vadMinSpeechMs" type="number" min="0" step="10" />
            </label>
          </div>
        </div>

        <div class="actions">
          <button type="button" :disabled="vadLoading" @click="handleVad">
            {{ vadLoading ? '处理中...' : '开始检测' }}
          </button>
        </div>

        <div class="status" v-if="vadTaskId || vadError">
          <p v-if="vadTaskId">任务 ID：{{ vadTaskId }}</p>
          <p v-if="vadStatus">状态：{{ vadStatus }}，进度：{{ (vadProgress * 100).toFixed(0) }}%</p>
          <p v-if="vadError" class="error">错误：{{ vadError }}</p>
        </div>

        <div v-if="vadResult.length" class="result-box">
          <h3>语音区间（采样率 16000Hz）</h3>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>开始样本</th>
                <th>结束样本</th>
                <th>开始时间 (s)</th>
                <th>结束时间 (s)</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(chunk, idx) in vadResult" :key="idx">
                <td>{{ idx + 1 }}</td>
                <td>{{ chunk.start }}</td>
                <td>{{ chunk.end }}</td>
                <td>{{ (chunk.start / 16000).toFixed(2) }}</td>
                <td>{{ (chunk.end / 16000).toFixed(2) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <!-- BGM 分离 -->
      <section v-else class="panel">
        <h2>BGM 分离</h2>
        <div class="form-grid">
          <div class="form-item full-width">
            <label>
              音频 / 视频文件：
              <input type="file" accept="audio/*,video/*" @change="onBgmFileChange" />
            </label>
          </div>

          <div class="form-item">
            <label>
              UVR 模型：
              <select v-model="bgmModelSize">
                <option value="UVR-MDX-NET-Inst_HQ_4">UVR-MDX-NET-Inst_HQ_4</option>
                <option value="UVR-MDX-NET-Inst_3">UVR-MDX-NET-Inst_3</option>
              </select>
            </label>
          </div>
        </div>

        <div class="actions">
          <button type="button" :disabled="bgmLoading" @click="handleBgmSeparation">
            {{ bgmLoading ? '处理中...' : '开始分离' }}
          </button>
        </div>

        <div class="status" v-if="bgmTaskId || bgmError">
          <p v-if="bgmTaskId">任务 ID：{{ bgmTaskId }}</p>
          <p v-if="bgmStatus">状态：{{ bgmStatus }}，进度：{{ (bgmProgress * 100).toFixed(0) }}%</p>
          <p v-if="bgmError" class="error">错误：{{ bgmError }}</p>
        </div>

        <div v-if="bgmResult" class="result-box">
          <h3>分离结果（哈希）</h3>
          <p>instrumental_hash: {{ bgmResult.instrumental_hash }}</p>
          <p>vocal_hash: {{ bgmResult.vocal_hash }}</p>

          <div v-if="bgmDownloadUrl" class="download">
            <a :href="bgmDownloadUrl" download="bgm_separation.zip">下载结果 ZIP</a>
            <p class="tip">ZIP 内通常包含伴奏和人声音轨两个文件。</p>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>

<style scoped>
.app-root {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  color: #1f2933;
  background: #f3f4f6;
}

.app-header {
  padding: 1.5rem 2rem 1rem;
  background: #111827;
  color: #f9fafb;
}

.app-header h1 {
  margin: 0;
  font-size: 1.5rem;
}

.subtitle {
  margin: 0.25rem 0 0.75rem;
  font-size: 0.9rem;
  color: #9ca3af;
}

.api-base {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  flex-wrap: wrap;
}

.api-base input {
  width: 260px;
}

.content {
  flex: 1;
  padding: 1.5rem 2rem 2rem;
}

.hero {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(0, 1.4fr);
  gap: 1.5rem;
  margin-bottom: 1.5rem;
  align-items: center;
}

.hero-text {
  padding: 1rem 0.75rem 1rem 0;
}

.hero-tag {
  display: inline-block;
  padding: 0.1rem 0.75rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.8);
  font-size: 0.75rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #6b7280;
  margin-bottom: 0.5rem;
}

.hero h2 {
  margin: 0;
  font-size: 1.8rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  color: #111827;
}

.hero-desc {
  margin-top: 0.6rem;
  font-size: 0.9rem;
  line-height: 1.7;
  color: #4b5563;
}

.hero-carousel {
  position: relative;
  border-radius: 0.75rem;
  overflow: hidden;
  background: radial-gradient(circle at top left, #fef3c7, #e5e7eb 40%, #f9fafb 70%);
  box-shadow: 0 20px 40px rgba(15, 23, 42, 0.18);
}

.carousel-window {
  position: relative;
  height: 260px;
}

.carousel-slide {
  position: absolute;
  inset: 0;
  opacity: 0;
  transform: translateX(8px) scale(1.02);
  transition: opacity 600ms ease, transform 600ms ease;
  display: flex;
  flex-direction: column;
}

.carousel-slide.active {
  opacity: 1;
  transform: translateX(0) scale(1);
}

.carousel-image {
  flex: 1;
  background-size: cover;
  background-position: center;
  filter: saturate(0.9) contrast(1.02);
}

.carousel-caption {
  padding: 0.6rem 0.9rem 0.7rem;
  background: linear-gradient(to right, rgba(17, 24, 39, 0.92), rgba(30, 64, 175, 0.85));
  color: #f9fafb;
}

.carousel-caption h3 {
  margin: 0 0 0.2rem;
  font-size: 0.95rem;
  font-weight: 600;
}

.carousel-caption p {
  margin: 0;
  font-size: 0.8rem;
  opacity: 0.95;
}

.carousel-nav {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 2rem;
  height: 2rem;
  border-radius: 999px;
  border: none;
  background: rgba(17, 24, 39, 0.55);
  color: #f9fafb;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.3rem;
}

.carousel-nav.prev {
  left: 0.4rem;
}

.carousel-nav.next {
  right: 0.4rem;
}

.carousel-nav:hover {
  background: rgba(15, 23, 42, 0.85);
}

.carousel-dots {
  position: absolute;
  bottom: 0.4rem;
  right: 0.7rem;
  display: flex;
  gap: 0.3rem;
}

.dot {
  width: 0.35rem;
  height: 0.35rem;
  border-radius: 999px;
  border: none;
  background: rgba(249, 250, 251, 0.7);
  cursor: pointer;
}

.dot.active {
  width: 0.8rem;
  background: #fbbf24;
}

.tabs {
  display: inline-flex;
  border-radius: 999px;
  background: #e5e7eb;
  padding: 0.25rem;
  margin-bottom: 1rem;
}

.tab-btn {
  border: none;
  background: transparent;
  padding: 0.4rem 1.2rem;
  border-radius: 999px;
  font-size: 0.9rem;
  cursor: pointer;
  color: #4b5563;
}

.tab-btn.active {
  background: #111827;
  color: #f9fafb;
}

.panel {
  background: #ffffff;
  border-radius: 0.75rem;
  padding: 1.25rem 1.5rem 1.5rem;
  box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
  animation: panel-fade-in 480ms ease;
}

.panel h2 {
  margin-top: 0;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.75rem 1rem;
  margin-bottom: 1rem;
}

.form-item {
  display: flex;
  flex-direction: column;
  font-size: 0.9rem;
}

.form-item.full-width {
  grid-column: 1 / -1;
}

label {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

input[type='text'],
input[type='number'],
input[type='file'],
select {
  padding: 0.35rem 0.5rem;
  border-radius: 0.375rem;
  border: 1px solid #d1d5db;
  font-size: 0.9rem;
}

.checkbox {
  flex-direction: row;
  align-items: center;
  gap: 0.4rem;
}

.actions {
  margin-bottom: 0.75rem;
}

button[type='button'] {
  padding: 0.4rem 1.1rem;
  border-radius: 999px;
  border: none;
  background: #111827;
  color: #f9fafb;
  font-size: 0.9rem;
  cursor: pointer;
}

button[type='button']:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.status {
  font-size: 0.85rem;
  color: #4b5563;
}

.error {
  color: #b91c1c;
}

.result-box {
  margin-top: 1rem;
  border-radius: 0.5rem;
  border: 1px solid #e5e7eb;
  padding: 0.75rem 0.9rem;
  background: #f9fafb;
}

.result-box pre {
  margin: 0.5rem 0 0;
  max-height: 360px;
  overflow: auto;
  white-space: pre-wrap;
  font-size: 0.85rem;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

th,
td {
  border: 1px solid #e5e7eb;
  padding: 0.35rem 0.5rem;
  text-align: center;
}

thead {
  background: #f3f4f6;
}

.download {
  margin-top: 0.5rem;
}

.download a {
  color: #1d4ed8;
  text-decoration: none;
}

.download a:hover {
  text-decoration: underline;
}

.tip {
  font-size: 0.8rem;
  color: #6b7280;
}

@keyframes panel-fade-in {
  from {
    opacity: 0;
    transform: translateY(6px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (max-width: 640px) {
  .app-header,
  .content {
    padding: 1rem;
  }

  .hero {
    grid-template-columns: minmax(0, 1fr);
  }
  .carousel-window {
    height: 210px;
  }
}
</style>
