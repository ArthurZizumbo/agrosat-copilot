<script setup lang="ts">
// Segmented A/B LLM switch (Gemini / Qwen). Reflects + drives the chat store's
// llmVariant via the parent through v-model:variant.

import type { LlmVariant } from "~/types/agent";

const props = defineProps<{
  variant: LlmVariant;
  disabled?: boolean;
}>();

const emit = defineEmits<{ (e: "update:variant", v: LlmVariant): void }>();

const { t } = useI18n();

const options: { value: LlmVariant; labelKey: string; short: string }[] = [
  { value: "gemini", labelKey: "llm.gemini", short: "Gemini" },
  { value: "qwen35", labelKey: "llm.qwen", short: "Qwen" },
];

function pick(v: LlmVariant) {
  if (v === props.variant || props.disabled) return;
  emit("update:variant", v);
}
</script>

<template>
  <div
    class="inline-flex items-center gap-0.5 rounded-md border border-[var(--color-border)] bg-[var(--color-surface-2)] p-0.5"
    role="group"
    :aria-label="t('chat.switch_llm')"
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
      :disabled="disabled"
      :title="t(opt.labelKey)"
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
