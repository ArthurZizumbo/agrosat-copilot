// TailwindCSS v4 config — AgroSatCopilot
// El theme principal se configura vía CSS en assets/css/main.css con @theme.
// Este archivo existe para tooling que aún espera tailwind.config.ts.

import type { Config } from "tailwindcss";

export default {
  content: [
    "./components/**/*.{vue,ts,tsx}",
    "./layouts/**/*.{vue,ts,tsx}",
    "./pages/**/*.{vue,ts,tsx}",
    "./plugins/**/*.{ts,tsx}",
    "./app.vue",
    "./error.vue",
    "./node_modules/@nuxt/ui-pro/**/*.{vue,ts}",
  ],
  theme: {
    extend: {
      colors: {
        agrosat: {
          primary: "#15803d",
          secondary: "#0891b2",
          accent: "#f59e0b",
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
