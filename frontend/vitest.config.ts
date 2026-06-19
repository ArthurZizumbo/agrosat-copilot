import { resolve } from "node:path";
import { defineConfig } from "vitest/config";

// Minimal Vitest config for unit tests that do not need the Nuxt runtime.
// The chat store reducer is plain Pinia + TypeScript, so the default `node`
// environment is enough. The `~` alias mirrors Nuxt's srcDir so store/type
// imports resolve in tests.
export default defineConfig({
  resolve: {
    alias: {
      "~": resolve(__dirname, "."),
      "@": resolve(__dirname, "."),
    },
  },
  test: {
    environment: "node",
    include: ["tests/unit/**/*.{test,spec}.ts"],
    globals: true,
  },
});
