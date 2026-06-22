<script setup lang="ts">
// A single transcript bubble (user or assistant) with optional citations.
//
// SECURITY: the assistant turn is LLM output and is rendered as markdown. The
// raw text is NEVER fed to v-html directly; it goes through `renderMarkdown`
// (marked -> isomorphic-dompurify) which strips <script>, event handlers and
// javascript: URLs. The user turn is the user's own text and stays plain
// `whitespace-pre-wrap` (no markdown) to keep the XSS surface minimal.

import { computed } from "vue";
import { renderMarkdown } from "~/utils/markdown";
import { useChatStore } from "~/stores/chat";
import type { ChatMessage } from "~/types/chat";

const props = defineProps<{ message: ChatMessage }>();
const { t } = useI18n();
const store = useChatStore();

const isAssistant = computed(() => props.message.role === "assistant");

// Sanitised HTML for the assistant turn only. Deterministic on server/client
// (isomorphic-dompurify), so it does not break SSR hydration.
const renderedHtml = computed(() =>
  isAssistant.value ? renderMarkdown(props.message.text) : "",
);

// Assistant turn still "thinking": it must be the turn CURRENTLY being streamed
// (the store's active assistant id) AND carry no reply text yet. Keying on the
// active id -- not merely on an empty body -- is what stops the spinner: `done`
// and `error` both clear `activeAssistantId`, so a turn that settled without a
// final answer (e.g. the reasoner exhausted its tool budget) no longer spins
// forever. Once any text streams, it is also no longer pending.
const isPending = computed(
  () =>
    isAssistant.value &&
    props.message.text.length === 0 &&
    store.activeAssistantId === props.message.id,
);

// The ReasoningCard renders above the reply when the turn has grounding OR is
// still pending (so the "thinking" placeholder appears immediately).
const showReasoning = computed(
  () =>
    isAssistant.value &&
    ((props.message.reasoning?.trim().length ?? 0) > 0 || isPending.value),
);

// The reply bubble only appears once there is text or citations to show.
const hasCitations = computed(
  () => (props.message.citations?.length ?? 0) > 0,
);
const showBubble = computed(
  () =>
    !isAssistant.value ||
    props.message.text.length > 0 ||
    hasCitations.value,
);
</script>

<template>
  <article
    class="flex flex-col gap-1"
    :class="message.role === 'user' ? 'items-end' : 'items-start'"
  >
    <span class="flex items-center gap-1 px-1 text-[0.6875rem] font-medium text-[var(--color-muted-fg)]">
      <UIcon
        :name="message.role === 'user' ? 'i-lucide-user' : 'i-lucide-sparkles'"
        class="size-3"
        aria-hidden="true"
      />
      {{ message.role === "user" ? t("chat.you") : t("chat.assistant") }}
    </span>

    <!-- Reasoning card: shown ABOVE the reply for the assistant turn. Carries
         the "thinking" placeholder while pending and the perceiver observation
         ("what the agent saw") once it lands. -->
    <ChatReasoningCard
      v-if="showReasoning"
      class="max-w-[90%]"
      :reasoning="message.reasoning"
      :pending="isPending"
    />

    <div
      v-if="showBubble"
      class="max-w-[90%] rounded-[var(--radius-lg)] px-3 py-2 text-sm"
      :class="
        message.role === 'user'
          ? 'rounded-tr-sm bg-agro-700 text-white'
          : 'rounded-tl-sm border border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-fg)]'
      "
    >
      <!-- Assistant: sanitised markdown (tables, code blocks). The HTML passed
           to v-html is ALWAYS sanitised by renderMarkdown; never the raw LLM
           text. -->
      <!-- eslint-disable-next-line vue/no-v-html -->
      <div
        v-if="isAssistant"
        class="markdown-body leading-relaxed"
        :aria-label="t('chat.assistant')"
        v-html="renderedHtml"
      />
      <!-- User: own text, plain, no markdown. -->
      <p v-else class="whitespace-pre-wrap leading-relaxed">{{ message.text }}</p>

      <ul
        v-if="message.citations && message.citations.length > 0"
        class="mt-2 space-y-1 border-t pt-2 text-left text-xs"
        :class="message.role === 'user' ? 'border-white/25' : 'border-[var(--color-border)]'"
        :aria-label="t('chat.citations')"
      >
        <li
          v-for="(cite, idx) in message.citations"
          :key="`${message.id}-cite-${idx}`"
          :class="message.role === 'user' ? 'text-white/85' : 'text-[var(--color-muted-fg)]'"
        >
          <span class="font-mono font-medium tabular-nums">[{{ idx + 1 }}]</span>
          {{ cite.source }}
          <template v-if="cite.parcel_id != null">
            · {{ t("chat.parcel") }} {{ cite.parcel_id }}
          </template>
          <template v-if="cite.dates && cite.dates.length">
            · {{ cite.dates.join(", ") }}
          </template>
        </li>
      </ul>
    </div>
  </article>
</template>

<style scoped>
/* Minimal, theme-aware markdown styling for the assistant bubble. Uses
   currentColor / border tokens so it works in light and dark mode (the bubble
   sets `text-[var(--color-fg)]`). Tailwind's `prose` is not assumed installed,
   so the essentials (tables, code, lists, links) are styled here. */
.markdown-body :deep(> :first-child) {
  margin-top: 0;
}

.markdown-body :deep(> :last-child) {
  margin-bottom: 0;
}

.markdown-body :deep(p) {
  margin: 0.5em 0;
}

.markdown-body :deep(h1),
.markdown-body :deep(h2),
.markdown-body :deep(h3),
.markdown-body :deep(h4) {
  margin: 0.75em 0 0.35em;
  font-weight: 600;
  line-height: 1.3;
}

.markdown-body :deep(h1) {
  font-size: 1.15em;
}

.markdown-body :deep(h2) {
  font-size: 1.08em;
}

.markdown-body :deep(h3) {
  font-size: 1em;
}

.markdown-body :deep(ul),
.markdown-body :deep(ol) {
  margin: 0.5em 0;
  padding-inline-start: 1.25em;
}

.markdown-body :deep(li) {
  margin: 0.2em 0;
}

.markdown-body :deep(a) {
  color: var(--color-agro-700, #2f7d4f);
  text-decoration: underline;
  text-underline-offset: 2px;
}

:global(.dark) .markdown-body :deep(a) {
  color: var(--color-agro-400, #6fcf97);
}

.markdown-body :deep(code) {
  border-radius: var(--radius-sm, 0.25rem);
  background: color-mix(in srgb, currentColor 10%, transparent);
  padding: 0.1em 0.35em;
  font-family: var(--font-mono, ui-monospace, monospace);
  font-size: 0.85em;
}

.markdown-body :deep(pre) {
  margin: 0.6em 0;
  overflow-x: auto;
  border-radius: var(--radius-md, 0.5rem);
  border: 1px solid var(--color-border);
  background: color-mix(in srgb, currentColor 6%, transparent);
  padding: 0.7em 0.85em;
}

.markdown-body :deep(pre code) {
  background: transparent;
  padding: 0;
  font-size: 0.82em;
  line-height: 1.5;
}

.markdown-body :deep(blockquote) {
  margin: 0.6em 0;
  border-inline-start: 3px solid var(--color-border);
  padding-inline-start: 0.75em;
  color: var(--color-muted-fg);
}

.markdown-body :deep(table) {
  display: block;
  width: max-content;
  max-width: 100%;
  margin: 0.6em 0;
  overflow-x: auto;
  border-collapse: collapse;
  font-size: 0.9em;
}

.markdown-body :deep(th),
.markdown-body :deep(td) {
  border: 1px solid var(--color-border);
  padding: 0.35em 0.6em;
  text-align: start;
}

.markdown-body :deep(th) {
  background: color-mix(in srgb, currentColor 8%, transparent);
  font-weight: 600;
}

.markdown-body :deep(hr) {
  margin: 0.8em 0;
  border: 0;
  border-top: 1px solid var(--color-border);
}

.markdown-body :deep(img) {
  max-width: 100%;
  height: auto;
}
</style>
