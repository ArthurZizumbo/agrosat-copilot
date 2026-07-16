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
const { llmVariant, isBusy, llmSwitchError } = storeToRefs(chatStore);
const { switchLlm } = useChat();

// Auto-dismiss the LLM-switch notice a few seconds after it appears (E12). The
// notice is transient: it only signals that persisting the choice failed (e.g.
// the on-prem host is down); the chat keeps working on the previous backend.
// The pending timer is tracked and cleared before re-arming, so a second failure
// gets its own full delay instead of being wiped by the first notice's timer.
const NOTICE_MS = 5000;
let noticeTimer: ReturnType<typeof setTimeout> | null = null;

function clearNoticeTimer() {
  if (noticeTimer !== null) {
    clearTimeout(noticeTimer);
    noticeTimer = null;
  }
}

watch(llmSwitchError, (msg) => {
  if (!import.meta.client) return;
  clearNoticeTimer();
  if (!msg) return;
  noticeTimer = setTimeout(() => {
    noticeTimer = null;
    chatStore.setLlmSwitchError(null);
  }, NOTICE_MS);
});

onBeforeUnmount(clearNoticeTimer);

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
      <div class="relative hidden sm:block">
        <!-- ClientOnly: llmVariant comes from localStorage (persisted store), which
             the server cannot know -> SSR would render the default and the client
             the persisted value, triggering a hydration class mismatch. Rendering
             this switch client-only removes the mismatch (the switch is a user
             gesture anyway, never needed at first paint). -->
        <ClientOnly>
          <ChatLlmSwitch :variant="llmVariant" :disabled="isBusy" @update:variant="onSwitchLlm" />
          <template #fallback>
            <div class="h-9 w-54 rounded-md bg-surface-2" />
          </template>
        </ClientOnly>
        <!-- Transient notice when persisting the switch failed (E12). -->
        <p
          v-if="llmSwitchError"
          role="status"
          class="absolute right-0 top-full z-40 mt-1 max-w-[16rem] rounded-[var(--radius-sm)] border border-amber-300 bg-amber-50 px-2 py-1 text-xs text-amber-800 shadow-[var(--shadow-panel)] dark:border-amber-500/40 dark:bg-amber-950/60 dark:text-amber-200"
        >
          {{ llmSwitchError }}
        </p>
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
