import { describe, expect, it } from "vitest";
import { renderMarkdown } from "~/utils/markdown";

// Exercises the REAL markdown pipeline (marked -> isomorphic-dompurify) that
// MessageBubble.vue feeds to `v-html`. This is the load-bearing security
// boundary: untrusted LLM text MUST be sanitised before it reaches the DOM.
//
// The markdown sources below are representative assistant turns (a GFM table and
// a fenced code block) plus the classic XSS vectors an LLM could echo back.

describe("renderMarkdown (assistant turn rendering)", () => {
  it("renders a GFM table into <table>/<th>/<td>", () => {
    const md = [
      "| Parcel | Crop | NDVI |",
      "| --- | --- | --- |",
      "| 10 | Meadow | 0.59 |",
      "| 11 | Vineyard | 0.71 |",
    ].join("\n");
    const html = renderMarkdown(md);
    expect(html).toContain("<table>");
    expect(html).toContain("<th>Crop</th>");
    expect(html).toContain("<td>Vineyard</td>");
  });

  it("renders a fenced code block into <pre><code>", () => {
    const md = "Here is the query:\n\n```sql\nSELECT * FROM parcels;\n```";
    const html = renderMarkdown(md);
    expect(html).toContain("<pre>");
    expect(html).toContain("<code");
    expect(html).toContain("SELECT * FROM parcels;");
  });

  it("keeps inline code and links but renders text safely", () => {
    const html = renderMarkdown("Use `NDVI` and see [docs](https://example.com).");
    expect(html).toContain("<code>NDVI</code>");
    expect(html).toContain('href="https://example.com"');
  });

  it("strips a <script> tag (XSS)", () => {
    const html = renderMarkdown("Hello <script>alert('xss')</script> world");
    expect(html).not.toContain("<script");
    expect(html).not.toContain("alert('xss')");
    expect(html).toContain("Hello");
  });

  it("strips an onerror event handler from a raw <img> tag (XSS)", () => {
    const html = renderMarkdown("<img src=x onerror=alert(1)>");
    // The <img> survives but with NO event handler attribute.
    expect(html).toContain("<img");
    expect(html.toLowerCase()).not.toContain("onerror");
  });

  it("strips a javascript: URL from a link (XSS)", () => {
    const html = renderMarkdown("[click me](javascript:alert(document.cookie))");
    expect(html.toLowerCase()).not.toContain("javascript:");
  });

  it("returns an empty string for empty input", () => {
    expect(renderMarkdown("")).toBe("");
  });
});
