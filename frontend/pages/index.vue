<script setup lang="ts">
// Main screen. The map fills the content slot of the dashboard shell. The map's
// imperative API (flyToDemoAoi / locateParcel) is registered by MapCanvas into
// the "mapCanvas" ref PROVIDED BY THE LAYOUT, so the sidebar and FindingCards
// can drive the map. The page itself only renders the map.

const { t } = useI18n();

useHead({ title: t("app.name") });
</script>

<template>
  <MapCanvas />
</template>
