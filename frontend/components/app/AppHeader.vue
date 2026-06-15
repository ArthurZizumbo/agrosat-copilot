<script setup lang="ts">
// Sticky app header: brand, centred basemap switcher, LLM A/B switch, theme
// toggle, locale selector. Emits sidebar/chat toggles for small viewports.

import { storeToRefs } from "pinia";
import { useChatStore } from "~/stores/chat";
import type { LlmVariant } from "~/types/agent";

const emit = defineEmits<{
  (e: "toggle-sidebar"): void;
  (e: "toggle-chat"): void;
}>();

const { t, locale, locales, setLocale } = useI18n();
const colorMode = useColorMode();
const chatStore = useChatStore();
const { llmVariant, isBusy } = storeToRefs(chatStore);
const { switchLlm } = useChat();

const isDark = computed({
  get: () => colorMode.value === "dark",
  set: (v) => {
    colorMode.preference = v ? "dark" : "light";
  },
});

const availableLocales = computed(() =>
  (locales.value as { code: string; name?: string }[]).map((l) => ({
    code: l.code,
    name: l.name ?? l.code.toUpperCase(),
  })),
);

async function onSwitchLlm(v: LlmVariant) {
  if (v === llmVariant.value || isBusy.value) return;
  await switchLlm(v);
}
</script>

<template>
  <header
    class="sticky top-0 z-30 flex h-14 items-center justify-between gap-2 border-b border-[var(--color-border)] bg-[var(--color-surface)] px-3"
  >
    <!-- Left: sidebar toggle + brand -->
    <div class="flex min-w-0 items-center gap-2">
      <button
        type="button"
        class="inline-flex size-9 items-center justify-center rounded-[var(--radius-sm)] text-[var(--color-muted-fg)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-fg)] xl:hidden"
        :aria-label="t('nav.toggle_tools')"
        @click="emit('toggle-sidebar')"
      >
        <UIcon name="i-lucide-panel-left" class="size-5" aria-hidden="true" />
      </button>
      <div class="flex min-w-0 items-center gap-2">
        <span
          class="inline-flex size-7 shrink-0 items-center justify-center rounded-[var(--radius-sm)] bg-agro-700 text-white"
          aria-hidden="true"
        >
          <UIcon name="i-lucide-satellite-dish" class="size-4" />
        </span>
        <span class="truncate text-sm font-bold tracking-tight text-[var(--color-fg)]">
          {{ t("app.name") }}
        </span>
      </div>
    </div>

    <!-- Center: basemap switch (hidden on small) -->
    <div class="hidden md:block">
      <MapBasemapSwitcher />
    </div>

    <!-- Right: LLM switch + theme + locale + chat toggle -->
    <div class="flex items-center gap-1.5">
      <div class="hidden sm:block">
        <ChatLlmSwitch :variant="llmVariant" :disabled="isBusy" @update:variant="onSwitchLlm" />
      </div>

      <button
        type="button"
        class="inline-flex size-9 items-center justify-center rounded-[var(--radius-sm)] text-[var(--color-muted-fg)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-fg)]"
        :aria-label="t('nav.toggle_theme')"
        :aria-pressed="isDark"
        @click="isDark = !isDark"
      >
        <ClientOnly>
          <UIcon :name="isDark ? 'i-lucide-moon' : 'i-lucide-sun'" class="size-5" aria-hidden="true" />
          <template #fallback>
            <UIcon name="i-lucide-sun" class="size-5" aria-hidden="true" />
          </template>
        </ClientOnly>
      </button>

      <label class="sr-only" for="locale-select">{{ t("nav.language") }}</label>
      <select
        id="locale-select"
        class="h-9 rounded-[var(--radius-sm)] border border-[var(--color-border)] bg-[var(--color-surface-2)] px-2 text-xs font-medium text-[var(--color-fg)] focus:outline-none"
        :value="locale"
        @change="setLocale(($event.target as HTMLSelectElement).value as typeof locale)"
      >
        <option v-for="l in availableLocales" :key="l.code" :value="l.code">
          {{ l.code.toUpperCase() }}
        </option>
      </select>

      <button
        type="button"
        class="inline-flex size-9 items-center justify-center rounded-[var(--radius-sm)] text-[var(--color-muted-fg)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-fg)] lg:hidden"
        :aria-label="t('chat.toggle')"
        @click="emit('toggle-chat')"
      >
        <UIcon name="i-lucide-message-square" class="size-5" aria-hidden="true" />
      </button>
    </div>
  </header>
</template>
