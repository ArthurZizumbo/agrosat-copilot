<script setup lang="ts">
// Chat switcher for the dock header: a compact dropdown listing the browser's
// chats with new / switch / rename / delete. State lives in `useChats`
// (localStorage list + per-chat transcripts); this is purely the UI.

import { useChats } from "~/composables/useChats";

const { t } = useI18n();
const { chats, currentChatId, createChat, switchChat, deleteChat, renameChat } =
  useChats();

const open = ref(false);
const editingId = ref<string | null>(null);
const editTitle = ref("");

const currentTitle = computed(
  () =>
    chats.value.find((c) => c.id === currentChatId.value)?.title ??
    t("chat.assistant_title"),
);

async function onNew() {
  await createChat();
  open.value = false;
}

function onSwitch(id: string) {
  switchChat(id);
  open.value = false;
}

function startRename(id: string, title: string) {
  editingId.value = id;
  editTitle.value = title;
  nextTick(() => {
    const el = document.getElementById(`chat-rename-${id}`);
    if (el instanceof HTMLInputElement) el.focus();
  });
}

function commitRename(id: string) {
  if (editingId.value !== id) return;
  renameChat(id, editTitle.value);
  editingId.value = null;
}

async function onDelete(id: string) {
  editingId.value = null;
  await deleteChat(id);
}
</script>

<template>
  <div class="relative">
    <!-- Trigger: current chat title -->
    <button
      type="button"
      class="flex min-w-0 items-center gap-1 rounded-[var(--radius-sm)] px-1.5 py-1 text-sm font-semibold text-[var(--color-fg)] hover:bg-[var(--color-surface-2)]"
      :aria-label="t('chat.chats')"
      :aria-expanded="open"
      @click="open = !open"
    >
      <span class="truncate max-w-[10rem]">{{ currentTitle }}</span>
      <UIcon
        name="i-lucide-chevron-down"
        class="size-3.5 shrink-0 text-[var(--color-muted-fg)] transition-transform"
        :class="{ 'rotate-180': open }"
        aria-hidden="true"
      />
    </button>

    <!-- Backdrop (click-outside to close) -->
    <div v-if="open" class="fixed inset-0 z-40" @click="open = false" />

    <!-- Panel -->
    <div
      v-if="open"
      class="absolute left-0 top-full z-50 mt-1 w-64 overflow-hidden rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)] shadow-lg"
      role="menu"
    >
      <button
        type="button"
        class="flex w-full items-center gap-2 border-b border-[var(--color-border)] px-3 py-2 text-sm font-medium text-agro-700 hover:bg-agro-50 dark:text-agro-400 dark:hover:bg-agro-900/20"
        @click="onNew"
      >
        <UIcon name="i-lucide-plus" class="size-4 shrink-0" aria-hidden="true" />
        {{ t("chat.new") }}
      </button>

      <ul class="max-h-72 overflow-y-auto py-1">
        <li v-if="chats.length === 0" class="px-3 py-2 text-xs text-[var(--color-muted-fg)]">
          {{ t("chat.empty") }}
        </li>
        <li
          v-for="c in chats"
          :key="c.id"
          class="group flex items-center gap-1 px-1.5 py-0.5"
        >
          <!-- Rename input -->
          <input
            v-if="editingId === c.id"
            :id="`chat-rename-${c.id}`"
            v-model="editTitle"
            type="text"
            class="min-w-0 flex-1 rounded-[var(--radius-sm)] border border-agro-600 bg-[var(--color-surface)] px-2 py-1 text-sm text-[var(--color-fg)] outline-none"
            @keyup.enter="commitRename(c.id)"
            @keyup.escape="editingId = null"
            @blur="commitRename(c.id)"
          />
          <!-- Switch button -->
          <button
            v-else
            type="button"
            class="flex min-w-0 flex-1 items-center gap-2 rounded-[var(--radius-sm)] px-2 py-1.5 text-left text-sm hover:bg-[var(--color-surface-2)]"
            :class="
              c.id === currentChatId
                ? 'font-semibold text-agro-700 dark:text-agro-400'
                : 'text-[var(--color-fg)]'
            "
            @click="onSwitch(c.id)"
          >
            <UIcon
              name="i-lucide-message-square"
              class="size-3.5 shrink-0 text-[var(--color-muted-fg)]"
              aria-hidden="true"
            />
            <span class="truncate">{{ c.title }}</span>
          </button>

          <!-- Row actions -->
          <button
            v-if="editingId !== c.id"
            type="button"
            class="shrink-0 rounded-[var(--radius-sm)] p-1 text-[var(--color-muted-fg)] opacity-0 hover:bg-[var(--color-surface-2)] hover:text-[var(--color-fg)] group-hover:opacity-100"
            :aria-label="t('chat.rename')"
            @click="startRename(c.id, c.title)"
          >
            <UIcon name="i-lucide-pencil" class="size-3.5" aria-hidden="true" />
          </button>
          <button
            v-if="editingId !== c.id"
            type="button"
            class="shrink-0 rounded-[var(--radius-sm)] p-1 text-[var(--color-muted-fg)] opacity-0 hover:bg-state-danger/10 hover:text-state-danger group-hover:opacity-100"
            :aria-label="t('chat.delete')"
            @click="onDelete(c.id)"
          >
            <UIcon name="i-lucide-trash-2" class="size-3.5" aria-hidden="true" />
          </button>
        </li>
      </ul>
    </div>
  </div>
</template>
