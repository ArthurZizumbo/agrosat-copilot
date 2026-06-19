<script setup lang="ts">
// Segmented A/B LLM switch (Gemini / Qwen). Reflects the chat store's llmVariant
// via v-model:variant.
//
// APORTE PENDIENTE: the team backend's `/chat` does NOT accept a per-request
// llm_variant (the reasoner is fixed by `settings.llm_variant_default`). When
// `server-fixed` is set the control renders disabled with a "server
// configuration" tooltip so the design is preserved without a misleading action.

import type { LlmVariant } from "~/types/agent";

const props = defineProps<{
  variant: LlmVariant;
  disabled?: boolean;
  /** The variant is fixed server-side; show a tooltip explaining why. */
  serverFixed?: boolean;
}>();

const emit = defineEmits<{ (e: "update:variant", v: LlmVariant): void }>();

const { t } = useI18n();

const options: { value: LlmVariant; labelKey: string; short: string }[] = [
  { value: "gemini", labelKey: "llm.gemini", short: "Gemini" },
  { value: "qwen35", labelKey: "llm.qwen", short: "Qwen" },
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
