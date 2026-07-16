<script setup lang="ts">
// Compact crop-classification model selector. Lets the user pin which model
// `classify_new_parcel` serves (voting3 / xgb / stacking5) instead of leaving the
// choice to the LLM. Reflects the chat store's `cropModel` via v-model:model.
// Unlike LlmSwitch, this IS wired end-to-end AND it is a HARD choice: the backend
// carries `ChatRequest.crop_model` on the tool context and `classify.run` serves
// it verbatim, ignoring the model argument the reasoner passed -- the LLM cannot
// opt out of the user's selection.

import type { CropModel } from "~/types/agent";

const props = defineProps<{
  /** The model the user actively pinned, or null when they never chose one. */
  model: CropModel | null;
  disabled?: boolean;
}>();

const emit = defineEmits<{ (e: "update:model", v: CropModel): void }>();

const { t } = useI18n();

const options: { value: CropModel; labelKey: string }[] = [
  { value: "voting3", labelKey: "crop_model.voting3" },
  { value: "xgb", labelKey: "crop_model.xgb" },
  { value: "stacking5", labelKey: "crop_model.stacking5" },
];

/** What to HIGHLIGHT. With no pin the champion is what the backend tool serves by
 *  default, so showing it as active is honest; the difference between "no pin" and
 *  "pinned voting3" is invisible here on purpose, and only matters to the backend
 *  (a pin overrides the reasoner, no pin leaves it free). */
const active = computed<CropModel>(() => props.model ?? "voting3");

function pick(v: CropModel) {
  if (v === active.value || props.disabled) return;
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
          active === opt.value
            ? 'bg-[var(--color-surface)] text-[var(--color-fg)] shadow-[var(--shadow-panel)]'
            : 'text-[var(--color-muted-fg)] hover:text-[var(--color-fg)]'
        "
        :aria-pressed="active === opt.value"
        :disabled="disabled"
        :title="t(opt.labelKey)"
        @click="pick(opt.value)"
      >
        {{ t(opt.labelKey) }}
      </button>
    </div>
  </div>
</template>
