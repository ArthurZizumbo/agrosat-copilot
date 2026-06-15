<script setup lang="ts">
// Conversational dock: header (title + WS status dot + active-area chip),
// transcript with rich findings, live plan stepper, tool activity, empty/skeleton
// states, and the composer. Owns the chat transport via useChat.
// Refactor of the legacy ChatPanel.vue (same store contract).

import { storeToRefs } from "pinia";
import { useChatStore } from "~/stores/chat";
import { useMapStore } from "~/stores/map";
import type { LlmVariant } from "~/types/agent";
import { demoFindings } from "~/utils/demoPreview";

const emit = defineEmits<{
  (e: "close"): void;
  (e: "locate", parcelId: number): void;
}>();

const { t } = useI18n();
const store = useChatStore();
const mapStore = useMapStore();
const { messages, currentPlan, toolCalls, findings, status, llmVariant } =
  storeToRefs(store);
const { previewActive } = storeToRefs(mapStore);

const { sendMessage, switchLlm, dispose } = useChat();

const isBusy = computed(
  () => status.value === "dispatching" || status.value === "streaming",
);
const isEmpty = computed(
  () => messages.value.length === 0 && !isBusy.value,
);

// WS status dot: connected (streaming), idle, error.
const wsStatus = computed(() => {
  if (status.value === "error") return { cls: "bg-state-danger", key: "ws.error" };
  if (status.value === "streaming") return { cls: "bg-state-success animate-pulse", key: "ws.live" };
  if (status.value === "dispatching") return { cls: "bg-amber-cta-500 animate-pulse", key: "ws.connecting" };
  return { cls: "bg-[var(--color-border-strong)]", key: "ws.idle" };
});

const transcript = ref<HTMLElement | null>(null);
watch(
  [messages, toolCalls, currentPlan],
  async () => {
    await nextTick();
    if (transcript.value) transcript.value.scrollTop = transcript.value.scrollHeight;
  },
  { deep: true },
);

async function onSubmit(text: string) {
  if (isBusy.value) return;
  mapStore.setPreviewActive(false);
  await sendMessage(text);
}

async function onSwitchLlm(variant: LlmVariant) {
  if (variant === llmVariant.value || isBusy.value) return;
  await switchLlm(variant);
}

function loadExample() {
  mapStore.setPreviewActive(true);
  store.loadPreview(t("chat.preview_answer"), demoFindings());
}

onBeforeUnmount(() => {
  dispose();
});
</script>

<template>
  <section
    class="flex h-full flex-col bg-[var(--color-surface)]"
    :aria-label="t('chat.panel_label')"
  >
    <!-- Header -->
    <header
      class="flex items-center justify-between gap-2 border-b border-[var(--color-border)] px-3 py-2.5"
    >
      <div class="flex min-w-0 items-center gap-2">
        <span
          class="size-2 shrink-0 rounded-full"
          :class="wsStatus.cls"
          :aria-label="t(wsStatus.key)"
          role="status"
        />
        <h2 class="truncate text-sm font-semibold text-[var(--color-fg)]">
          {{ t("chat.assistant_title") }}
        </h2>
      </div>
      <div class="flex items-center gap-1.5">
        <ChatLlmSwitch
          :variant="llmVariant"
          :disabled="isBusy"
          @update:variant="onSwitchLlm"
        />
        <button
          type="button"
          class="inline-flex size-9 items-center justify-center rounded-[var(--radius-sm)] text-[var(--color-muted-fg)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-fg)] lg:hidden"
          :aria-label="t('chat.close')"
          @click="emit('close')"
        >
          <UIcon name="i-lucide-x" class="size-5" aria-hidden="true" />
        </button>
      </div>
    </header>

    <!-- Preview banner -->
    <div
      v-if="previewActive"
      class="flex items-center gap-1.5 border-b border-amber-cta-600/30 bg-amber-cta-50 px-3 py-1.5 text-xs text-amber-cta-700 dark:bg-amber-cta-600/15 dark:text-amber-cta-300"
      role="note"
    >
      <UIcon name="i-lucide-info" class="size-3.5" aria-hidden="true" />
      {{ t("chat.preview_badge") }}
    </div>

    <!-- Transcript -->
    <div
      ref="transcript"
      class="flex-1 space-y-3 overflow-y-auto p-3"
      role="log"
      aria-live="polite"
      :aria-busy="isBusy"
    >
      <!-- Empty state -->
      <div
        v-if="isEmpty"
        class="flex h-full flex-col items-center justify-center gap-3 px-4 text-center"
      >
        <span
          class="inline-flex size-12 items-center justify-center rounded-full bg-agro-50 text-agro-700 dark:bg-agro-900/40 dark:text-agro-400"
        >
          <UIcon name="i-lucide-messages-square" class="size-6" aria-hidden="true" />
        </span>
        <p class="text-sm font-medium text-[var(--color-fg)]">{{ t("chat.empty_title") }}</p>
        <p class="max-w-xs text-xs text-[var(--color-muted-fg)]">{{ t("chat.empty") }}</p>
        <button
          type="button"
          class="mt-1 inline-flex items-center gap-1.5 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--color-fg)] hover:border-agro-600"
          @click="loadExample"
        >
          <UIcon name="i-lucide-eye" class="size-3.5" aria-hidden="true" />
          {{ t("chat.see_example") }}
        </button>
      </div>

      <template v-else>
        <ChatMessageBubble
          v-for="msg in messages"
          :key="msg.id"
          :message="msg"
        />

        <ChatPlanStepper :steps="currentPlan" />
        <ChatToolActivity :calls="toolCalls" />

        <!-- Findings as rich cards -->
        <div v-if="findings.length > 0" class="space-y-2">
          <p
            class="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-[var(--color-muted-fg)]"
          >
            <UIcon name="i-lucide-sprout" class="size-3.5" aria-hidden="true" />
            {{ t("chat.findings") }}
            <span class="font-mono tabular-nums">({{ findings.length }})</span>
          </p>
          <ChatFindingCard
            v-for="f in findings"
            :key="`${f.citation.tool_call_id}-${f.parcel_id}`"
            :finding="f"
            @locate="emit('locate', $event)"
          />
        </div>

        <!-- Thinking skeleton -->
        <div v-if="isBusy && currentPlan.length === 0" class="space-y-2" aria-hidden="true">
          <div class="h-3 w-2/3 animate-pulse rounded bg-[var(--color-surface-2)]" />
          <div class="h-3 w-1/2 animate-pulse rounded bg-[var(--color-surface-2)]" />
        </div>
        <p
          v-if="isBusy"
          class="flex items-center gap-1.5 text-xs italic text-[var(--color-muted-fg)]"
        >
          <UIcon name="i-lucide-loader-circle" class="size-3.5 animate-spin" aria-hidden="true" />
          {{ t("chat.thinking") }}
        </p>

        <!-- Error -->
        <p
          v-if="status === 'error'"
          class="flex items-center gap-1.5 rounded-[var(--radius-md)] border border-state-danger/30 bg-state-danger/10 px-3 py-2 text-sm text-state-danger"
          role="alert"
        >
          <UIcon name="i-lucide-triangle-alert" class="size-4 shrink-0" aria-hidden="true" />
          {{ t("errors.generic") }}
        </p>
      </template>
    </div>

    <!-- Composer -->
    <ChatComposer
      :disabled="isBusy"
      :show-suggestions="isEmpty"
      @submit="onSubmit"
    />
  </section>
</template>
