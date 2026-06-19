<script setup lang="ts">
// Compact rows of tool activity: tool name (mono), status (icon + text, never
// colour-only) and duration in mono tabular-nums.

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
</script>

<template>
  <ul
    v-if="calls.length > 0"
    class="space-y-1"
    :aria-label="t('chat.tools')"
  >
    <li
      v-for="call in calls"
      :key="call.call_id"
      class="flex items-center justify-between gap-2 rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface)] px-2.5 py-1.5 text-xs"
    >
      <span class="flex min-w-0 items-center gap-1.5">
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
    </li>
  </ul>
</template>
