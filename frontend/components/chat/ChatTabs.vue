<script setup lang="ts">
// In-app chat tabs (US-080): one tab per backend session. Switching tabs swaps
// the transcript (reloaded from the server) AND the map AOI (each tab keeps its
// own drawn zone). New/close create/delete the underlying session.

import { storeToRefs } from "pinia";
import { useSessionsStore } from "~/stores/sessions";

const { t } = useI18n();
const store = useSessionsStore();
const { tabs, activeId } = storeToRefs(store);
const { createSession, switchSession, closeSession } = useSessions();

// Serialise tab actions: each does async API + map/chat side effects.
const busy = ref(false);

async function run(action: () => Promise<unknown>) {
  if (busy.value) return;
  busy.value = true;
  try {
    await action();
  } finally {
    busy.value = false;
  }
}

const onNew = () => run(() => createSession());
const onSelect = (id: string) =>
  id === activeId.value ? undefined : run(() => switchSession(id));
const onClose = (id: string) => run(() => closeSession(id));
</script>

<template>
  <div
    class="flex items-center gap-1 overflow-x-auto border-b border-[var(--color-border)] px-2 py-1.5"
    role="tablist"
    :aria-label="t('chat.tabs_label')"
  >
    <button
      v-for="tab in tabs"
      :key="tab.id"
      type="button"
      role="tab"
      :aria-selected="tab.id === activeId"
      :disabled="busy"
      class="group flex max-w-[10rem] shrink-0 items-center gap-1 rounded-[var(--radius-sm)] px-2 py-1 text-xs font-medium"
      :class="
        tab.id === activeId
          ? 'bg-agro-700 text-white'
          : 'bg-[var(--color-surface-2)] text-[var(--color-fg)] hover:bg-[var(--color-surface)]'
      "
      @click="onSelect(tab.id)"
    >
      <span class="truncate">{{ tab.title }}</span>
      <span
        v-if="tabs.length > 1"
        class="inline-flex size-4 shrink-0 items-center justify-center rounded-full opacity-60 hover:bg-black/20 hover:opacity-100"
        role="button"
        :aria-label="t('chat.close_chat')"
        @click.stop="onClose(tab.id)"
      >
        <UIcon name="i-lucide-x" class="size-3" aria-hidden="true" />
      </span>
    </button>

    <button
      type="button"
      class="inline-flex size-7 shrink-0 items-center justify-center rounded-[var(--radius-sm)] text-[var(--color-muted-fg)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-fg)] disabled:opacity-50"
      :aria-label="t('chat.new_chat')"
      :disabled="busy"
      @click="onNew"
    >
      <UIcon name="i-lucide-plus" class="size-4" aria-hidden="true" />
    </button>
  </div>
</template>
