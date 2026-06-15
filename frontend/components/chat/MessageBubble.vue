<script setup lang="ts">
// A single transcript bubble (user or assistant) with optional citations.

import type { ChatMessage } from "~/types/chat";

defineProps<{ message: ChatMessage }>();
const { t } = useI18n();
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
    <div
      class="max-w-[90%] rounded-[var(--radius-lg)] px-3 py-2 text-sm"
      :class="
        message.role === 'user'
          ? 'rounded-tr-sm bg-agro-700 text-white'
          : 'rounded-tl-sm border border-[var(--color-border)] bg-[var(--color-surface)] text-[var(--color-fg)]'
      "
    >
      <p class="whitespace-pre-wrap leading-relaxed">{{ message.text }}</p>

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
