// Flat config (ESLint v9). El proyecto usa ESLint 9, que requiere flat config
// (`eslint.config.*`) y ya NO soporta `.eslintrc` ni el flag `--ext`.
//
// Cubre .ts/.mjs/.js con el parser de TypeScript (typescript-eslint) y las reglas
// recomendadas de JS + TS. Los .vue se excluyen del lint (requieren
// vue-eslint-parser, no instalado); su chequeo de tipos lo cubre `pnpm typecheck`
// (vue-tsc). Cuando se agregue `@nuxt/eslint`, este archivo se reemplaza por el
// config generado por Nuxt (que ya trae el parser de Vue).
import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: [
      ".nuxt/**",
      ".output/**",
      "dist/**",
      "node_modules/**",
      "coverage/**",
      "**/*.vue",
    ],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["**/*.{js,mjs,ts}"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
    },
    rules: {
      // El frontend usa auto-imports de Nuxt; no-undef es ruido aqui (lo valida
      // vue-tsc). Las vars no usadas se reportan como warning, no como error.
      "no-undef": "off",
      "@typescript-eslint/no-unused-vars": [
        "warn",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      "@typescript-eslint/no-explicit-any": "off",
    },
  },
);
