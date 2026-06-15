<script setup lang="ts">
// The orchestrator's live plan rendered as a vertical stepper.

defineProps<{ steps: string[] }>();
const { t } = useI18n();
</script>

<template>
  <section
    v-if="steps.length > 0"
    class="rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface-2)] p-3"
    :aria-label="t('chat.plan')"
  >
    <p
      class="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-[var(--color-muted-fg)]"
    >
      <UIcon name="i-lucide-list-checks" class="size-3.5" aria-hidden="true" />
      {{ t("chat.plan") }}
    </p>
    <ol class="space-y-0">
      <li
        v-for="(step, idx) in steps"
        :key="`step-${idx}`"
        class="relative flex gap-2.5 pb-2.5 last:pb-0"
      >
        <span
          v-if="idx < steps.length - 1"
          class="absolute left-[0.6875rem] top-5 h-full w-px bg-[var(--color-border)]"
          aria-hidden="true"
        />
        <span
          class="z-10 mt-0.5 inline-flex size-[1.375rem] shrink-0 items-center justify-center rounded-full bg-agro-700 font-mono text-[0.625rem] font-semibold tabular-nums text-white"
        >
          {{ idx + 1 }}
        </span>
        <span class="text-sm leading-tight text-[var(--color-fg)]">{{ step }}</span>
      </li>
    </ol>
  </section>
</template>
