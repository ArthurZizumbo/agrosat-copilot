<script setup lang="ts">
const { t } = useI18n();
useHead({
  titleTemplate: (title) => (title ? `${title} · ${t("app.name")}` : t("app.name")),
  htmlAttrs: { lang: useI18n().locale.value },
});
</script>

<template>
  <NuxtLayout>
    <NuxtPage />
  </NuxtLayout>
</template>
