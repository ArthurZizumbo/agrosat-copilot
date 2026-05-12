<script setup lang="ts">
const { t } = useI18n();
</script>

<template>
  <main class="min-h-screen flex items-center justify-center p-8">
    <section class="max-w-2xl text-center space-y-4">
      <h1 class="text-4xl font-bold">{{ t("app.name") }}</h1>
      <p class="text-lg text-gray-600 dark:text-gray-400">{{ t("app.tagline") }}</p>
    </section>
  </main>
</template>
