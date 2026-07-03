<script setup lang="ts">
// Compact crop-classification model selector. Lets the user pin the model the
// reasoner forwards to `classify_new_parcel` (voting3 / xgb / stacking5),
// instead of leaving the choice to the LLM. Reflects the chat store's
// `cropModel` via v-model:model. Unlike LlmSwitch, this IS wired end-to-end:
// the backend accepts `ChatRequest.crop_model` and injects a system turn.

import type { CropModel } from "~/types/agent";

const props = defineProps<{
  model: CropModel;
  disabled?: boolean;
}>();

const emit = defineEmits<{ (e: "update:model", v: CropModel): void }>();

const { t } = useI18n();

const options: { value: CropModel; labelKey: string }[] = [
  { value: "voting3", labelKey: "crop_model.voting3" },
  { value: "xgb", labelKey: "crop_model.xgb" },
  { value: "stacking5", labelKey: "crop_model.stacking5" },
];

function pick(v: CropModel) {
  if (v === props.model || props.disabled) return;
  emit("update:model", v);
}
</script>

<template>
  <div
    class="inline-flex items-center gap-1"
    role="group"
    :aria-label="t('crop_model.label')"
  >
    <label
      class="text-xs font-medium text-[var(--color-muted-fg)]"
      :title="t('crop_model.hint')"
    >
      {{ t("crop_model.label") }}
    </label>
    <div
      class="inline-flex items-center gap-0.5 rounded-md border border-[var(--color-border)] bg-[var(--color-surface-2)] p-0.5"
    >
      <button
        v-for="opt in options"
        :key="opt.value"
        type="button"
        class="inline-flex min-h-8 items-center rounded-[var(--radius-sm)] px-2 text-xs font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-50"
        :class="
          model === opt.value
            ? 'bg-[var(--color-surface)] text-[var(--color-fg)] shadow-[var(--shadow-panel)]'
            : 'text-[var(--color-muted-fg)] hover:text-[var(--color-fg)]'
        "
        :aria-pressed="model === opt.value"
        :disabled="disabled"
        :title="t(opt.labelKey)"
        @click="pick(opt.value)"
      >
        {{ t(opt.labelKey) }}
      </button>
    </div>
  </div>
</template>
