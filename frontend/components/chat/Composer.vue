<script setup lang="ts">
// Message composer: area-context chip above the input, suggested-prompt chips
// when empty, auto-grow textarea and a send button.

defineProps<{ disabled?: boolean; showSuggestions?: boolean }>();
const emit = defineEmits<{ (e: "submit", text: string): void }>();

const { t } = useI18n();
const draft = ref("");

const suggestions = computed<string[]>(() => [
  t("chat.suggest_1"),
  t("chat.suggest_2"),
  t("chat.suggest_3"),
]);

function submit() {
  const text = draft.value.trim();
  if (!text) return;
  emit("submit", text);
  draft.value = "";
}

function useSuggestion(text: string) {
  draft.value = text;
  submit();
}
</script>

<template>
  <form
    class="flex flex-col gap-2 border-t border-[var(--color-border)] bg-[var(--color-surface)] p-3"
    @submit.prevent="submit"
  >
    <!-- Area context chip -->
    <ChatAreaChip dismissible />

    <!-- Suggested prompts when empty -->
    <div
      v-if="showSuggestions"
      class="flex flex-wrap gap-1.5"
      :aria-label="t('chat.suggestions')"
    >
      <button
        v-for="(s, idx) in suggestions"
        :key="idx"
        type="button"
        class="rounded-full border border-[var(--color-border)] bg-[var(--color-surface-2)] px-2.5 py-1 text-xs text-[var(--color-muted-fg)] transition-colors hover:border-agro-600 hover:text-[var(--color-fg)]"
        :disabled="disabled"
        @click="useSuggestion(s)"
      >
        {{ s }}
      </button>
    </div>

    <div class="flex items-end gap-2">
      <label class="sr-only" for="chat-input">{{ t("chat.input_label") }}</label>
      <textarea
        id="chat-input"
        v-model="draft"
        rows="2"
        class="min-h-[2.75rem] flex-1 resize-none rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-sm text-[var(--color-fg)] placeholder:text-[var(--color-muted-fg)] focus:border-agro-600 focus:outline-none"
        :placeholder="t('chat.placeholder')"
        :disabled="disabled"
        @keydown.enter.exact.prevent="submit"
      />
      <button
        type="submit"
        class="inline-flex size-11 shrink-0 items-center justify-center rounded-[var(--radius-md)] bg-agro-700 text-white transition-colors hover:bg-agro-800 disabled:cursor-not-allowed disabled:opacity-50"
        :disabled="disabled || !draft.trim()"
        :aria-label="t('chat.send')"
      >
        <UIcon name="i-lucide-send-horizontal" class="size-5" aria-hidden="true" />
      </button>
    </div>
  </form>
</template>
