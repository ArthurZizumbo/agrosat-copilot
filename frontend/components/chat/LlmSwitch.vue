<script setup lang="ts">
// Segmented per-session reasoner switch (Gemini / Qwen / Qwen-VL). Reflects the
// chat store's llmVariant via v-model:variant.
//
// E12: the switch is REAL. Picking a backend emits `update:variant`, which the
// parent forwards to `useChat.switchLlm` -> `POST /llm/switch` (persisted on the
// session; the next `/chat` builds the matching backend). The on-prem variants
// (`qwen-onprem` / `qwen-vl`) are reachable only behind the demo VM tunnel; a
// failed switch reverts and surfaces a toast (the chat never breaks). The
// `disabled` prop only suppresses changes mid-stream; `serverFixed` is kept for
// deployments that pin the backend server-side (then a tooltip explains why).

import type { LlmVariant } from "~/types/agent";

const props = defineProps<{
  variant: LlmVariant;
  disabled?: boolean;
  /** The variant is fixed server-side; show a tooltip explaining why. */
  serverFixed?: boolean;
}>();

const emit = defineEmits<{ (e: "update:variant", v: LlmVariant): void }>();

const { t } = useI18n();

// 1:1 with the backend variant tags (types/agent.ts LlmVariant): three options.
const options: { value: LlmVariant; labelKey: string; short: string }[] = [
  { value: "gemini", labelKey: "llm.gemini", short: "Gemini" },
  { value: "qwen-onprem", labelKey: "llm.qwen", short: "Qwen" },
  { value: "qwen-vl", labelKey: "llm.qwen_vl", short: "Qwen-VL" },
];

const isDisabled = computed(() => props.disabled || props.serverFixed);

function tooltipFor(labelKey: string): string {
  return props.serverFixed ? t("llm.server_fixed") : t(labelKey);
}

function pick(v: LlmVariant) {
  if (v === props.variant || isDisabled.value) return;
  emit("update:variant", v);
}
</script>

<template>
  <div
    class="inline-flex items-center gap-0.5 rounded-md border border-[var(--color-border)] bg-[var(--color-surface-2)] p-0.5"
    role="group"
    :aria-label="t('chat.switch_llm')"
    :title="serverFixed ? t('llm.server_fixed') : undefined"
  >
    <button
      v-for="opt in options"
      :key="opt.value"
      type="button"
      class="inline-flex min-h-9 items-center gap-1.5 rounded-[var(--radius-sm)] px-2.5 text-xs font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-50"
      :class="
        variant === opt.value
          ? 'bg-[var(--color-surface)] text-[var(--color-fg)] shadow-[var(--shadow-panel)]'
          : 'text-[var(--color-muted-fg)] hover:text-[var(--color-fg)]'
      "
      :aria-pressed="variant === opt.value"
      :disabled="isDisabled"
      :title="tooltipFor(opt.labelKey)"
      @click="pick(opt.value)"
    >
      <span
        class="size-1.5 rounded-full"
        :class="variant === opt.value ? 'bg-agro-600' : 'bg-[var(--color-border-strong)]'"
        aria-hidden="true"
      />
      {{ opt.short }}
    </button>
  </div>
</template>
