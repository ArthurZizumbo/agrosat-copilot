<script setup lang="ts">
// Collapsible rows of tool activity. The summary shows the tool name (mono),
// status (icon + text, never colour-only) and an optional one-line summary.
// Expanding reveals the input (`args`) and output (`result`) as formatted JSON.
//
// A11y: built on native <details>/<summary>, so keyboard toggle (Enter/Space),
// focus management and the expanded/collapsed semantics come for free. The
// <summary> carries an aria-label; the chevron animation respects
// prefers-reduced-motion.

import type { TrackedToolCall } from "~/types/chat";

defineProps<{ calls: TrackedToolCall[] }>();
const { t } = useI18n();

function statusMeta(s: TrackedToolCall["status"]): {
  icon: string;
  label: string;
  cls: string;
} {
  if (s === "running")
    return {
      icon: "i-lucide-loader-circle",
      label: t("chat.tool_running"),
      cls: "text-amber-cta-700 dark:text-amber-cta-300",
    };
  if (s === "ok")
    return {
      icon: "i-lucide-check-circle-2",
      label: t("chat.tool_ok"),
      cls: "text-state-success dark:text-agro-400",
    };
  return {
    icon: "i-lucide-x-circle",
    label: t("chat.tool_failed"),
    cls: "text-state-danger",
  };
}

/** Pretty-print a JSON payload; empty objects/undefined render as a dash. */
function formatJson(value: unknown): string {
  if (value == null) return "—";
  if (typeof value === "object" && Object.keys(value).length === 0) return "{}";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function hasContent(value: Record<string, unknown> | undefined): boolean {
  return value != null && Object.keys(value).length > 0;
}
</script>

<template>
  <ul
    v-if="calls.length > 0"
    class="space-y-1"
    :aria-label="t('chat.tools')"
  >
    <li v-for="call in calls" :key="call.call_id">
      <details
        class="group rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface)] text-xs"
      >
        <summary
          class="flex cursor-pointer list-none items-center justify-between gap-2 rounded-[var(--radius-sm)] px-2.5 py-1.5 outline-none focus-visible:ring-2 focus-visible:ring-agro-600 focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--color-surface)]"
          :aria-label="t('chat.tool_details', { tool: call.tool })"
        >
          <span class="flex min-w-0 items-center gap-1.5">
            <UIcon
              name="i-lucide-chevron-right"
              class="chevron size-3.5 shrink-0 text-[var(--color-muted-fg)] group-open:rotate-90"
              aria-hidden="true"
            />
            <UIcon
              :name="statusMeta(call.status).icon"
              class="size-3.5 shrink-0"
              :class="[
                statusMeta(call.status).cls,
                call.status === 'running' ? 'animate-spin' : '',
              ]"
              aria-hidden="true"
            />
            <span class="truncate font-mono text-[var(--color-fg)]">{{ call.tool }}</span>
          </span>
          <span class="flex shrink-0 items-center gap-2">
            <span :class="statusMeta(call.status).cls">{{ statusMeta(call.status).label }}</span>
            <span
              v-if="call.summary"
              class="max-w-[8rem] truncate font-mono tabular-nums text-[var(--color-muted-fg)]"
            >
              {{ call.summary }}
            </span>
          </span>
        </summary>

        <div class="space-y-2 border-t border-[var(--color-border)] px-2.5 py-2">
          <div>
            <p class="mb-1 font-medium uppercase tracking-wide text-[var(--color-muted-fg)]">
              {{ t("chat.tool_input") }}
            </p>
            <pre
              class="overflow-x-auto rounded-[var(--radius-sm)] bg-[color-mix(in_srgb,currentColor_6%,transparent)] p-2 font-mono text-[0.6875rem] leading-relaxed text-[var(--color-fg)]"
            >{{ formatJson(call.args) }}</pre>
          </div>
          <div>
            <p class="mb-1 font-medium uppercase tracking-wide text-[var(--color-muted-fg)]">
              {{ t("chat.tool_output") }}
            </p>
            <pre
              v-if="hasContent(call.result)"
              class="overflow-x-auto rounded-[var(--radius-sm)] bg-[color-mix(in_srgb,currentColor_6%,transparent)] p-2 font-mono text-[0.6875rem] leading-relaxed text-[var(--color-fg)]"
            >{{ formatJson(call.result) }}</pre>
            <p
              v-else
              class="italic text-[var(--color-muted-fg)]"
            >
              {{ statusMeta(call.status).label }}
            </p>
          </div>
        </div>
      </details>
    </li>
  </ul>
</template>

<style scoped>
/* Native marker hidden in favour of the chevron icon. */
summary::-webkit-details-marker {
  display: none;
}

.chevron {
  transition: transform 150ms ease;
}

@media (prefers-reduced-motion: reduce) {
  .chevron {
    transition: none;
  }
}
</style>
