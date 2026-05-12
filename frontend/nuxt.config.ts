// Nuxt 4 SSR config — AgroSatCopilot
// Regla §13 CLAUDE.md: i18n it/es/en obligatorio.
// Regla §3.5: MapLibre GL + deck.gl (OSS), nada de Mapbox.

export default defineNuxtConfig({
  compatibilityDate: "2026-05-01",
  devtools: { enabled: true },
  ssr: true,

  modules: [
    "@nuxt/ui-pro",
    "@nuxtjs/i18n",
    "@pinia/nuxt",
  ],

  css: ["~/assets/css/main.css"],

  i18n: {
    defaultLocale: "it",
    strategy: "prefix_except_default",
    locales: [
      { code: "it", iso: "it-IT", file: "it.json", name: "Italiano" },
      { code: "es", iso: "es-ES", file: "es.json", name: "Español" },
      { code: "en", iso: "en-US", file: "en.json", name: "English" },
    ],
    langDir: "locales",
    detectBrowserLanguage: {
      useCookie: true,
      cookieKey: "agrosat-i18n",
      redirectOn: "root",
    },
  },

  runtimeConfig: {
    public: {
      apiBaseUrl: process.env.NUXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000",
      clerkPublishableKey: process.env.NUXT_PUBLIC_CLERK_PUBLISHABLE_KEY ?? "",
      mapTilerKey: process.env.NUXT_PUBLIC_MAPTILER_KEY ?? "",
    },
  },

  typescript: {
    strict: true,
    typeCheck: false,
  },

  nitro: {
    preset: process.env.NITRO_PRESET ?? "node-server",
  },
});
