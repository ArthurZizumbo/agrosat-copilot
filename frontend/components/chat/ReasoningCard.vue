<script setup lang="ts">
// Per-turn "reasoning" card shown ABOVE the assistant reply, mimicking the
// reasoning/thinking blocks of mainstream AI chats. It unifies two formerly
// separate UI bits into a single small, collapsible card:
//   1. the "thinking" placeholder, while the turn waits for the first reply
//      delta (amber accent, spinner header);
//   2. the perceiver observation, "what the agent saw" (Be My Eyes), once it is
//      attached to the turn (agro accent, expandable body).
//
// Collapsed by default. Built on native <details>/<summary>, so keyboard toggle
// (Enter/Space), focus and the expanded/collapsed semantics come for free; the
// chevron + body transitions respect prefers-reduced-motion. The component is
// purely presentational: it owns no store state.

const props = defineProps<{
  /** Perceiver grounding ("what the agent saw"); absent while still pending. */
  reasoning?: string;
  /** The turn has no reply text yet (still "thinking"). */
  pending: boolean;
}>();

const { t } = useI18n();

// While pending with no grounding yet, show the amber "thinking" header; once
// the perceiver observation lands, switch to the agro "first analysis" header.
const hasReasoning = computed(
  () => (props.reasoning?.trim().length ?? 0) > 0,
);
const isThinking = computed(() => props.pending && !hasReasoning.value);
</script>

<template>
  <details
    class="reasoning-card group rounded-[var(--radius-sm)] border text-xs"
    :class="
      isThinking
        ? 'border-amber-cta-600/30 bg-amber-cta-50 dark:bg-amber-cta-600/10'
        : 'border-agro-600/30 bg-agro-50 dark:bg-agro-900/20'
    "
  >
    <summary
      class="flex cursor-pointer list-none items-center gap-1.5 rounded-[var(--radius-sm)] px-2.5 py-1.5 outline-none focus-visible:ring-2 focus-visible:ring-agro-600 focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--color-surface)]"
      :class="{ 'pointer-events-none': isThinking }"
      :aria-label="isThinking ? t('chat.thinking') : t('chat.observation')"
    >
      <UIcon
        v-if="isThinking"
        name="i-lucide-loader-circle"
        class="size-3.5 shrink-0 animate-spin text-amber-cta-700 dark:text-amber-cta-300"
        aria-hidden="true"
      />
      <UIcon
        v-else
        name="i-lucide-eye"
        class="size-3.5 shrink-0 text-agro-700 dark:text-agro-400"
        aria-hidden="true"
      />
      <span
        class="font-medium uppercase tracking-wide"
        :class="
          isThinking
            ? 'text-amber-cta-700 dark:text-amber-cta-300'
            : 'text-agro-700 dark:text-agro-400'
        "
      >
        {{ isThinking ? t("chat.thinking") : t("chat.observation") }}
      </span>
      <UIcon
        v-if="!isThinking"
        name="i-lucide-chevron-right"
        class="chevron ml-auto size-3.5 shrink-0 text-[var(--color-muted-fg)] group-open:rotate-90"
        aria-hidden="true"
      />
    </summary>

    <div
      v-if="hasReasoning"
      class="reasoning-body border-t border-agro-600/20 px-2.5 py-2"
    >
      <p class="whitespace-pre-wrap leading-relaxed text-[var(--color-fg)]">
        {{ reasoning }}
      </p>
    </div>
  </details>
</template>

<style scoped>
/* Native marker hidden in favour of the chevron icon. */
summary::-webkit-details-marker {
  display: none;
}

.chevron {
  transition: transform 150ms ease;
}

/* Subtle reveal of the body when expanded. */
.reasoning-card[open] .reasoning-body {
  animation: reasoning-reveal 160ms ease;
}

@keyframes reasoning-reveal {
  from {
    opacity: 0;
    transform: translateY(-2px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@media (prefers-reduced-motion: reduce) {
  .chevron {
    transition: none;
  }

  .reasoning-card[open] .reasoning-body {
    animation: none;
  }
}
</style>
