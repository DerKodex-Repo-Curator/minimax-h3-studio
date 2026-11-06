<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import ActionStrip from './ActionStrip.vue'
import UpscaleMenu from './UpscaleMenu.vue'
import ModelPreview from './ModelPreview.vue'
import TranscriptResult from './TranscriptResult.vue'
import { shortModel } from '../models'
import { PhCube, PhArrowCounterClockwise, PhDownloadSimple, PhFilmStrip, PhImage, PhMusicNotes, PhSpeakerHigh, PhStop, PhTrash, PhUserFocus, PhWarningCircle } from '@phosphor-icons/vue'

const props = defineProps({ job: Object, now: Number, upscale: Object })
const emit = defineEmits(['cancel', 'remove', 'reuse', 'use-ref', 'use-frame', 'use-audio', 'use-3d', 'use-edit', 'use-voice', 'upscale'])

const STAGES = { preprocessing: 'Preparing audio', synthesizing: 'Generating speech', transcribing: 'Transcribing audio', loading: 'Loading model', composing: 'Writing the song', encoding: 'Reading the prompt', sampling: 'Generating picture and sound', decoding: 'Decoding video', saving: 'Saving file', reading: 'Reading the video', upscaling: 'Upscaling' }
const MODES = { tts: 'Speech', asr: 'Transcription', i23d: '3D', t2v: 'Text', i2v: 'First frame', fl2v: 'First + last', r2v: 'Reference', t2i: 'Picture', i2i: 'Image edit', t2m: 'Music', upscale: 'Upscale' }

// Sampling dominates the run time, so it gets most of the bar.
const SPANS = { loading: [0, 0], encoding: [0, 0.05], sampling: [0.05, 0.86], decoding: [0.86, 0.98], saving: [0.98, 1] }
// Pictures: decoding a single image is instant, so sampling gets nearly all of the bar.
const PICTURE_SPANS = { loading: [0, 0], encoding: [0, 0.1], sampling: [0.1, 0.97], decoding: [0.97, 0.99], saving: [0.99, 1] }
// Music: the text encoder writes the acoustic tokens first, which takes most of the time.
const MUSIC_SPANS = { loading: [0, 0], composing: [0, 0.72], sampling: [0.72, 0.96], decoding: [0.96, 0.99], saving: [0.99, 1] }
// SeedVR2: one step per temporal chunk; its VAE encode and decode take most of the time.
const UPSCALE_SPANS = { loading: [0, 0], reading: [0, 0.25], upscaling: [0.25, 0.32], decoding: [0.32, 0.97], saving: [0.97, 1] }
// FlashVSR streams the clip, decoding and encoding as it goes, so its chunks cover the whole run.
const FLASHVSR_SPANS = { loading: [0, 0.03], upscaling: [0.03, 0.98], saving: [0.98, 1] }
// RelayVSR: one step on every 15th frame, then the Flash decoder writes every frame.
// Encoding the large output takes about half of the run and reports no progress of its own.
const RELAYVSR_SPANS = { loading: [0, 0], reading: [0, 0.03], upscaling: [0.03, 0.17], decoding: [0.17, 0.46], saving: [0.46, 1] }
const UPSCALER_SPANS = { flashvsr: FLASHVSR_SPANS, relayvsr: RELAYVSR_SPANS }
const is3D = computed(() => props.job.mode === 'i23d')
const MESH_STAGES = { preprocessing: 'Preparing the object', structure: 'Generating structure', shape: 'Generating shape', shape_detail: 'Refining geometry', texture: 'Generating materials', meshing: 'Baking textures' }
const MESH_SPANS = { loading: [0, 0], preprocessing: [0, .08], structure: [.08, .18], shape: [.18, .35], shape_detail: [.35, .62], texture: [.62, .85], meshing: [.85, .99], saving: [.99, 1] }
const isPicture = computed(() => ['t2i', 'i2i'].includes(props.job.mode))
const isSpeech = computed(() => props.job.mode === 'tts')
const isASR = computed(() => props.job.mode === 'asr')
const isNativeAudio = computed(() => isSpeech.value || isASR.value)
const isMusic = computed(() => props.job.mode === 't2m')
const isUpscale = computed(() => props.job.mode === 'upscale')
const overall = computed(() => {
  const spans = is3D.value ? MESH_SPANS : isPicture.value ? PICTURE_SPANS : isMusic.value ? MUSIC_SPANS
    : isUpscale.value ? (UPSCALER_SPANS[props.job.upscaler] || UPSCALE_SPANS) : SPANS
  const span = spans[props.job.stage]
  return span ? span[0] + (span[1] - span[0]) * (props.job.progress || 0) : 0
})

const elapsed = computed(() => {
  const j = props.job
  if (!j.started_at) return null
  return Math.max(0, Math.round((j.finished_at || props.now / 1000) - j.started_at))
})

const frames = computed(() => props.job.result?.frames)
const clipLength = computed(() => {
  if (isMusic.value || isSpeech.value || isASR.value) {
    if (!props.job.duration && !props.job.result?.seconds) return '-'
    const s = Math.round(props.job.result?.seconds ?? props.job.duration)
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
  }
  return `${frames.value ? (frames.value / 24).toFixed(1) : +(props.job.duration || 0).toFixed(1)}s`
})
// Per upscaler: short sides above the video's ('1080p' -> 1080) or factors within the limit ('2x' -> 2);
// picking an upscaler and a target queues the job.
const upscalers = computed(() => {
  if (!props.job.video_url) return []
  const short = Math.min(props.job.width, props.job.height)
  const fits = (t) => t.endsWith('x') ? parseInt(t) * short <= props.upscale.relayvsr_max_short_side : parseInt(t) > short
  return (props.upscale?.upscalers || [])
    .map((u) => ({ ...u, targets: (u.targets || props.upscale.targets).filter(fits) }))
    .filter((u) => u.targets.length)
})
const created = computed(() => new Date(props.job.created_at * 1000).toLocaleString('en-US', {
  month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit',
}))

// Show the picture section only; the labels are the same on every job.
const scene = computed(() => {
  const m = props.job.prompt.match(/(?:integrated_multimodal_description|detailed_description):\s*([\s\S]*?)(\n\s*\n\s*overall_soundscape:|$)/)
  return (m ? m[1] : props.job.prompt).trim()
})

// The prompt is clamped to three lines; offer a toggle only when it actually overflows.
const sceneEl = ref(null)
const expanded = ref(false)
const clamped = ref(false)
let observer
function measure() {
  const el = sceneEl.value
  if (el && !expanded.value) clamped.value = el.scrollHeight > el.clientHeight + 1
}
onMounted(() => {
  measure()
  observer = new ResizeObserver(measure)
  observer.observe(sceneEl.value)
})
onBeforeUnmount(() => observer?.disconnect())
</script>

<template>
  <article class="card" :class="job.status">
    <div class="screen" :class="{ music: isMusic || isSpeech, transcription: isASR && job.status === 'done' }" :style="{ aspectRatio: is3D ? '1 / 1' : isASR && job.status === 'done' ? 'auto' : (isMusic || isNativeAudio) ? '16 / 6' : `${job.width} / ${job.height}` }">
      <TranscriptResult v-if="isASR && job.status === 'done'" :transcript="job.result?.transcript" />
      <ModelPreview v-else-if="is3D && job.model_3d_url" :src="job.model_3d_url" :poster="job.thumb_url" />
      <div v-else-if="(isMusic || isSpeech) && job.status === 'done' && job.audio_url" class="player">
        <PhSpeakerHigh v-if="isSpeech" :size="34" weight="duotone" /><PhMusicNotes v-else :size="34" weight="duotone" />
        <audio :src="job.audio_url" controls :preload="isSpeech ? 'metadata' : 'none'" />
      </div>
      <a v-else-if="job.status === 'done' && job.picture_url" :href="job.picture_url" target="_blank" rel="noopener">
        <img class="picture" :src="job.picture_url" :alt="job.prompt" loading="lazy" />
      </a>
      <video
        v-else-if="job.status === 'done' && job.video_url"
        :src="job.video_url"
        :poster="job.thumb_url"
        controls
        preload="none"
        playsinline
      />
      <template v-else>
        <img v-if="job.image_url || job.ref_urls?.length" class="ghost" :src="job.image_url || job.ref_urls[0]" alt="" />
        <div class="state">
          <template v-if="job.status === 'running'">
            <span class="rec"><i />REC</span>
            <span class="stage">{{ (is3D && MESH_STAGES[job.stage]) || STAGES[job.stage] || job.stage }}</span>
            <span class="track" :class="{ indeterminate: isNativeAudio }"><span :style="{ transform: `scaleX(${overall})` }" /></span>
          </template>
          <span v-else-if="job.status === 'queued'" class="stage">Queued, #{{ job.queue_position }}</span>
          <span v-else-if="job.status === 'failed'" class="stage failed"><PhWarningCircle :size="18" />Generation failed</span>
          <span v-else class="stage muted">Canceled</span>
        </div>
      </template>
    </div>

    <div class="body">
      <div class="prompt">
        <p ref="sceneEl" class="scene" :class="{ open: expanded }">{{ scene }}</p>
        <button v-if="clamped || expanded" class="more" @click="expanded = !expanded">
          {{ expanded ? 'Show less' : 'Show more' }}
        </button>
      </div>

      <details v-if="job.lyrics" class="lyrics">
        <summary>Lyrics</summary>
        <pre>{{ job.lyrics }}</pre>
      </details>

      <dl class="specs mono">
        <div><dt>Mode</dt><dd>{{ isSpeech ? ({ generate: 'Speech', clone: 'Voice clone', design: 'Voice design' }[job.audio_options?.speech_mode] || 'Speech') : MODES[job.mode] || job.mode }}<template v-if="job.mode === 'r2v'"><template v-if="job.refs.length || !job.ref_videos?.length"> ×{{ job.refs.length }}</template><template v-if="job.ref_videos?.length"> + {{ job.ref_videos.length }} video</template></template><PhImage v-else-if="job.image" :size="12" /><PhSpeakerHigh v-if="job.guide_audio" :size="12" :title="`Guide audio from ${job.audio_start || 0} s`" /></dd></div>
        <div v-if="!isMusic && !is3D && !isNativeAudio"><dt>Size</dt><dd>{{ job.width }}×{{ job.height }}</dd></div>
        <div v-if="!isPicture && !is3D"><dt>Length</dt><dd>{{ clipLength }}</dd></div>
        <div v-if="is3D"><dt>Detail</dt><dd>{{ job.mesh_options?.resolution }} · {{ (job.mesh_options?.texture_size || 2048) / 1024 }}K</dd></div>
        <div v-if="is3D && job.result?.faces"><dt>Faces</dt><dd>{{ job.result.faces.toLocaleString() }}</dd></div>
        <div><dt>Time</dt><dd>{{ elapsed === null ? '-' : `${elapsed}s` }}</dd></div>
      </dl>

      <p v-if="job.models" class="models mono" :title="[job.models.unet, job.models.lora].filter(Boolean).join('\n')">
        {{ shortModel(job.models.unet) }}<template v-if="job.models.lora"> + {{ shortModel(job.models.lora) }}<template v-if="(job.models.lora_strength ?? 1) !== 1"> ×{{ job.models.lora_strength }}</template></template><template v-if="job.sampler"> · {{ job.models.sampler }}</template><template v-if="job.steps"> · {{ job.models.steps }} steps</template>
      </p>

      <details v-if="job.status === 'failed' && job.error" class="error">
        <summary>Show error</summary>
        <pre>{{ job.error }}</pre>
      </details>

      <footer class="foot">
        <span class="meta mono" :title="`seed ${job.seed}`">{{ [job.preset, created].filter(Boolean).join(' · ') }}</span>
        <ActionStrip class="actions">
          <button v-if="job.status === 'queued' || job.status === 'running'" class="btn danger small" @click="emit('cancel', job)">
            <PhStop :size="13" weight="fill" />Cancel
          </button>
          <template v-else>
            <a v-for="(url, format) in job.transcript_urls || {}" :key="format" class="btn small" :href="url" :download="`transcript_${job.id}.${format === 'words' ? 'words.json' : format}`"><PhDownloadSimple :size="14" />{{ format === 'words' ? 'Words JSON' : format.toUpperCase() }}</a>
            <button v-if="isSpeech && job.audio_url && job.duration <= 30" class="btn small" @click="emit('use-voice', job)"><PhSpeakerHigh :size="14" />Clone voice</button>
            <a v-if="job.model_3d_url" class="btn small" :href="job.model_3d_url" :download="`pixal3d_${job.id}.glb`"><PhDownloadSimple :size="14" />GLB</a>
            <a v-if="job.video_url" class="btn small" :href="job.video_url" :download="`h3_${job.id}.mp4`">
              <PhDownloadSimple :size="14" />Download
            </a>
            <a v-if="job.audio_url" class="btn small" :href="job.audio_url" :download="`${isSpeech ? 'speech' : 'music'}_${job.id}.${isSpeech ? 'wav' : 'mp3'}`">
              <PhDownloadSimple :size="14" />Download
            </a>
            <button v-if="job.audio_url" class="btn small" title="Use as the guide audio of a video" @click="emit('use-audio', job)">
              <PhFilmStrip :size="14" />Video
            </button>
            <template v-if="job.picture_url">
              <a class="btn icon small" :href="job.picture_url" :download="`${job.image_model || 'moss'}_${job.id}.png`" aria-label="Download" title="Download">
                <PhDownloadSimple :size="14" />
              </a>
              <button class="btn small" title="Edit this picture with Klein or Qwen" @click="emit('use-edit', job)"><PhImage :size="14" />Edit</button>
              <button class="btn small" title="Turn this picture into a 3D model" @click="emit('use-3d', job)"><PhCube :size="14" />3D</button>
              <button class="btn small" title="Use as a reference image" @click="emit('use-ref', job)"><PhUserFocus :size="14" />Reference</button>
              <button class="btn small" title="Use as the first frame" @click="emit('use-frame', job)"><PhFilmStrip :size="14" />Frame</button>
            </template>
            <UpscaleMenu v-if="upscalers.length" :upscalers="upscalers" @select="(upscaler, target) => emit('upscale', job, upscaler, target)" />
            <button v-if="!isUpscale" class="btn small" @click="emit('reuse', job)"><PhArrowCounterClockwise :size="14" />Reuse</button>
            <button class="btn icon small danger" aria-label="Delete" @click="emit('remove', job)"><PhTrash :size="14" /></button>
          </template>
        </ActionStrip>
      </footer>
    </div>
  </article>
</template>

<style scoped>
.card {
  background: var(--surface); border-radius: var(--r-card); box-shadow: var(--shadow);
  overflow: hidden; display: flex; flex-direction: column; min-width: 0;
}
.screen { position: relative; background: var(--screen); max-height: 420px; width: 100%; }
.screen video, .picture { width: 100%; height: 100%; display: block; object-fit: contain; background: var(--screen); }
.screen > a { display: block; width: 100%; height: 100%; }
.player {
  position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 14px; padding: 16px; color: #dfe1e6;
}
.player audio { width: min(100%, 420px); }
.lyrics summary { cursor: pointer; color: var(--accent); font-size: 12px; }
.lyrics pre {
  margin: 6px 0 0; font: 12px/1.5 var(--font); color: var(--text-2); white-space: pre-wrap;
  max-height: 220px; overflow: auto;
}
.ghost { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; opacity: .28; filter: blur(1px); }
.state {
  position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 10px; color: #dfe1e6; padding: 16px;
}
.stage { font-size: 13px; display: inline-flex; align-items: center; gap: 6px; }
.stage.failed { color: #ff8d90; }
.rec {
  position: absolute; top: 12px; left: 12px; display: inline-flex; align-items: center; gap: 6px;
  font: 600 11px/1 var(--font-mono); letter-spacing: .08em; color: #f2f3f5;
}
.rec i { width: 8px; height: 8px; border-radius: 50%; background: var(--live); animation: tally 1.4s ease-in-out infinite; }
@keyframes tally { 50% { opacity: .25; } }
.track { width: min(70%, 320px); height: 3px; border-radius: 3px; background: rgba(255, 255, 255, .14); overflow: hidden; }
.track span { display: block; height: 100%; background: var(--accent); transform-origin: left; transition: transform .5s ease; }

.body { min-width: 0; padding: 14px 16px 12px; display: flex; flex-direction: column; gap: 12px; flex: 1; }
.scene {
  margin: 0; font-size: 13px; line-height: 1.5; color: var(--text);
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
  white-space: pre-line;
}
.scene.open { display: block; -webkit-line-clamp: unset; }
.more {
  margin-top: 2px; padding: 0; border: 0; background: none; cursor: pointer;
  font: inherit; font-size: 12px; color: var(--accent);
}
.more:hover { text-decoration: underline; }
.specs { display: grid; grid-template-columns: repeat(4, auto); justify-content: start; gap: 4px 18px; margin: 0; }
.specs div { display: flex; flex-direction: column; }
.specs dt { font-size: 10.5px; color: var(--muted); font-family: var(--font); }
.specs dd { margin: 0; font-size: 12px; display: inline-flex; align-items: center; gap: 4px; }
.models {
  margin: -4px 0 0; font-size: 11px; color: var(--muted);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.error summary { cursor: pointer; color: var(--live); font-size: 12px; }
.error pre { white-space: pre-wrap; font-size: 11px; color: var(--live); max-height: 200px; overflow: auto; margin: 6px 0 0; }
.foot { display: flex; flex-direction: column; align-items: stretch; gap: 8px; margin-top: auto; min-width: 0; }
.meta { font-size: 11px; color: var(--muted); white-space: nowrap; }
.actions { width: 100%; min-width: 0; }
.small { padding: 4px 11px; font-size: 12px; }
.icon.small { padding: 5px; }
</style>

<style scoped>
.screen.transcription { max-height: none; }
.indeterminate span { width: 35%; animation: native-progress 1.5s ease-in-out infinite; transform: none !important; }
@keyframes native-progress { 0% { margin-left: 0; } 50% { margin-left: 65%; } 100% { margin-left: 0; } }
</style>
