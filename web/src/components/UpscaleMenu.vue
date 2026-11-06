<script setup>
import { nextTick, onBeforeUnmount, ref } from 'vue'
import { PhArrowsOut } from '@phosphor-icons/vue'
defineProps({ upscalers: Array })
const emit = defineEmits(['select'])
const button = ref(null)
const panel = ref(null)
const open = ref(false)
const position = ref({})
function close() {
  open.value = false
  document.removeEventListener('pointerdown', outside)
  document.removeEventListener('keydown', escape)
  window.removeEventListener('scroll', close, true)
  window.removeEventListener('resize', close)
}
function outside(event) {
  if (!button.value?.contains(event.target) && !panel.value?.contains(event.target)) close()
}
function escape(event) {
  if (event.key === 'Escape') { close(); button.value?.focus({ preventScroll: true }) }
}
async function toggle() {
  if (open.value) { close(); return }
  position.value = { visibility: 'hidden' }
  open.value = true
  await nextTick()
  if (!open.value || !button.value || !panel.value) return
  const anchor = button.value.getBoundingClientRect()
  const size = panel.value.getBoundingClientRect()
  position.value = {
    left: `${Math.max(12, Math.min(anchor.right - size.width, window.innerWidth - size.width - 12))}px`,
    top: `${anchor.top > size.height + 12 ? anchor.top - size.height - 6 : anchor.bottom + 6}px`,
    visibility: 'visible',
  }
  panel.value.querySelector('button')?.focus({ preventScroll: true })
  document.addEventListener('pointerdown', outside)
  document.addEventListener('keydown', escape)
  window.addEventListener('scroll', close, true)
  window.addEventListener('resize', close)
}
function pick(upscaler, target) { close(); emit('select', upscaler, target) }
onBeforeUnmount(close)
</script>
<template>
  <div class="upscale">
    <button ref="button" class="btn small" :aria-expanded="open" @click="toggle"><PhArrowsOut :size="14" />Upscale</button>
    <Teleport to="body">
      <div v-if="open" ref="panel" class="upscale-menu" role="group" aria-label="Upscale options" :style="position">
        <div v-for="u in upscalers" :key="u.id" class="menu-row">
          <span class="menu-label">{{ u.label }}</span>
          <button v-for="t in u.targets" :key="t" class="btn small" @click="pick(u.id, t)">{{ t }}</button>
        </div>
      </div>
    </Teleport>
  </div>
</template>
<style scoped>
.small { padding: 4px 11px; font-size: 12px; }
.upscale-menu { position: fixed; z-index: 30; visibility: hidden; max-width: calc(100vw - 24px); display: flex; flex-direction: column; gap: 8px; padding: 12px; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; box-shadow: var(--shadow); }
.menu-row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.menu-label { font-size: 12px; color: var(--text-2); min-width: 96px; }
</style>
