<script setup lang="ts">
// Dashboard app shell: header + (sidebar | content | chat) + status bar.
// Responsive behaviour:
//   >=1280 (xl): 3 columns, sidebar collapsible to an icon rail.
//   1024-1280 (lg): sidebar as icon rail, map + chat.
//   768-1024 (md): chat as slide-over, sidebar as drawer, map full.
//   <768: single column, chat as bottom-sheet, tools in a drawer.
//
// The page content (map) is the default slot; the page wires demo/locate to the
// map via provide/inject of the MapCanvas ref. Header/sidebar/chat live here so
// every route shares the shell.

const { t } = useI18n();

// Mobile/tablet overlay state.
const sidebarOpen = ref(false);
const chatOpen = ref(false);
const sidebarCollapsed = ref(false);

// Chat bottom-sheet expansion on <768.
const sheetExpanded = ref(false);

function closeOverlays() {
  sidebarOpen.value = false;
  chatOpen.value = false;
}

// MapCanvas registers its imperative API here on mount. The layout is an
// ANCESTOR of both the sidebar and the page (<slot/> = MapCanvas), so the
// provider must live here — a provide() inside the page would not reach the
// sidebar via inject (inject only sees ancestors).
interface MapApi {
  flyToDemoAoi: () => void;
  locateParcel: (id: number) => void;
}
const mapApi = ref<MapApi | null>(null);
provide("mapCanvas", mapApi);

function onDemo() {
  mapApi.value?.flyToDemoAoi();
  sidebarOpen.value = false;
}
function onLocate(parcelId: number) {
  mapApi.value?.locateParcel(parcelId);
}
</script>

<template>
  <div class="flex h-dvh min-h-dvh flex-col overflow-hidden bg-[var(--color-bg)] text-[var(--color-fg)]">
    <AppHeader
      @toggle-sidebar="sidebarOpen = !sidebarOpen"
      @toggle-chat="chatOpen = !chatOpen"
    />

    <div class="relative flex min-h-0 flex-1">
      <!-- Sidebar: rail/expanded on >=xl; drawer overlay below xl -->
      <div class="hidden xl:block">
        <AppSidebar
          :collapsed="sidebarCollapsed"
          @demo="onDemo"
          @toggle-collapse="sidebarCollapsed = !sidebarCollapsed"
        />
      </div>
      <!-- lg: permanent icon rail -->
      <div class="hidden lg:block xl:hidden">
        <AppSidebar :collapsed="true" @demo="onDemo" />
      </div>

      <!-- Sidebar drawer (<lg) -->
      <Transition name="fade">
        <div
          v-if="sidebarOpen"
          class="absolute inset-0 z-40 bg-black/40 lg:hidden"
          @click="sidebarOpen = false"
        />
      </Transition>
      <Transition name="slide-left">
        <div v-if="sidebarOpen" class="absolute inset-y-0 left-0 z-50 lg:hidden">
          <AppSidebar @demo="onDemo" />
        </div>
      </Transition>

      <!-- Map / content -->
      <main class="relative min-w-0 flex-1" :aria-label="t('nav.map')">
        <slot />
      </main>

      <!-- Chat dock: permanent column >=lg -->
      <div class="hidden w-[25rem] shrink-0 border-l border-[var(--color-border)] lg:block">
        <ChatDock @locate="onLocate" />
      </div>

      <!-- Chat slide-over (md) / bottom-sheet (<md) -->
      <Transition name="fade">
        <div
          v-if="chatOpen"
          class="absolute inset-0 z-40 bg-black/40 lg:hidden"
          @click="chatOpen = false"
        />
      </Transition>
      <Transition name="slide-right">
        <div
          v-if="chatOpen"
          class="absolute z-50 lg:hidden md:inset-y-0 md:right-0 md:w-[24rem]
                 inset-x-0 bottom-0 md:bottom-auto"
          :class="sheetExpanded ? 'top-0' : 'top-1/3 md:top-0'"
        >
          <div class="flex h-full flex-col bg-[var(--color-surface)] shadow-[var(--shadow-pop)]">
            <!-- Bottom-sheet drag handle (mobile) -->
            <button
              type="button"
              class="flex w-full items-center justify-center py-1.5 md:hidden"
              :aria-label="t('chat.expand')"
              @click="sheetExpanded = !sheetExpanded"
            >
              <span class="h-1 w-10 rounded-full bg-[var(--color-border-strong)]" aria-hidden="true" />
            </button>
            <div class="min-h-0 flex-1">
              <ChatDock @close="chatOpen = false" @locate="onLocate" />
            </div>
          </div>
        </div>
      </Transition>
    </div>

    <AppStatusBar />
  </div>
</template>

<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
.slide-left-enter-active,
.slide-left-leave-active,
.slide-right-enter-active,
.slide-right-leave-active {
  transition: transform 0.25s ease;
}
.slide-left-enter-from,
.slide-left-leave-to {
  transform: translateX(-100%);
}
.slide-right-enter-from,
.slide-right-leave-to {
  transform: translateY(100%);
}
@media (min-width: 768px) {
  .slide-right-enter-from,
  .slide-right-leave-to {
    transform: translateX(100%);
  }
}
</style>
