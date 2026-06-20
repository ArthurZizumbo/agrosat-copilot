// Markdown rendering for untrusted LLM output.
//
// SECURITY (load-bearing): the assistant text comes from the LLM and is NOT
// trusted. It may contain `<script>`, `onerror=`, `javascript:` URLs, etc. The
// pipeline is ALWAYS parse -> sanitize -> v-html, never raw LLM text into
// v-html.
//
// - `marked` parses GFM (tables, fenced code blocks) and returns an HTML string
//   synchronously (`async: false`). Its legacy `sanitize` option is removed in
//   v12+, so sanitisation MUST be done externally.
// - `isomorphic-dompurify` runs DOMPurify on both the Node server (SSR) and the
//   browser, avoiding `window is not defined` during SSR and producing
//   deterministic HTML, so server and client renders match (no hydration drift).
//
// Only the assistant turn is rendered as markdown; the user turn stays plain
// `whitespace-pre-wrap` to shrink the XSS surface.

import DOMPurify from "isomorphic-dompurify";
import { marked } from "marked";

// GFM with tables/code; `breaks: true` maps single newlines to <br> (chat-like).
marked.use({ gfm: true, breaks: true });

/**
 * Render untrusted markdown to sanitised HTML safe for `v-html`.
 *
 * The result NEVER contains `<script>`, event handlers (`onerror`, `onclick`,
 * ...) or `javascript:` URLs: DOMPurify strips them. This is the only function
 * allowed to feed `v-html`; callers must pass the raw LLM text here first.
 *
 * @param src Raw markdown from the LLM (untrusted).
 * @returns Sanitised HTML string.
 */
export function renderMarkdown(src: string): string {
  if (!src) return "";
  const dirty = marked.parse(src, { async: false }) as string;
  return DOMPurify.sanitize(dirty);
}
