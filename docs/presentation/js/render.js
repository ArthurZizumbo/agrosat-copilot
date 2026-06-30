/* ============================================================================
   Motor de render de la presentacion AgroSatCopilot.
   El CONTENIDO vive en content/es.json y content/en.json (solo texto + figuras).
   Este motor construye los <section> de Reveal.js desde el idioma activo, y el
   switch de idioma re-renderiza desde el otro JSON. Asi, corregir un texto =
   editar una linea de JSON, sin tocar el HTML.
   ============================================================================ */
(function () {
  "use strict";

  // Escapa texto plano a HTML seguro (los JSON traen unicode normal).
  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // Texto "rich": escapa todo pero re-permite negrita/cursiva (<strong>, <em>,
  // <b>, <i>, <br>) que el contenido usa para enfatizar. Nada mas se interpreta.
  function rich(s) {
    return esc(s)
      .replace(/&lt;(\/?)(strong|em|b|i)&gt;/g, "<$1$2>")
      .replace(/&lt;br\s*\/?&gt;/g, "<br>");
  }

  function attr(name, val) {
    return val ? ` ${name}="${esc(val)}"` : "";
  }

  // Normaliza la ruta de una figura: si el JSON trae solo el nombre
  // (data_pipeline.png) le antepone assets/figs/. Si ya trae ruta, la respeta.
  function figUrl(img) {
    if (!img) return "";
    return img.indexOf("/") >= 0 ? img : `assets/figs/${img}`;
  }

  // --- Renderizadores por layout -------------------------------------------

  function kicker(s) {
    return s.kicker ? `<div class="kicker">${esc(s.kicker)}</div>` : "";
  }
  function title(s, tag) {
    tag = tag || "h2";
    return s.title ? `<${tag}>${esc(s.title)}</${tag}>` : "";
  }
  function paras(arr) {
    return (arr || []).map((p) => `<p>${rich(p)}</p>`).join("");
  }
  function bullets(arr) {
    if (!arr || !arr.length) return "";
    return `<ul class="ulist">${arr.map((b) => `<li>${rich(b)}</li>`).join("")}</ul>`;
  }

  function cards(items, cols) {
    const inner = (items || [])
      .map((it) => {
        const v = it.variant && it.variant !== "normal" ? ` ${it.variant.trim()}` : "";
        return `<div class="card${v}">` +
          (it.title ? `<h3>${rich(it.title)}</h3>` : "") +
          (it.body ? `<p>${rich(it.body)}</p>` : "") + `</div>`;
      })
      .join("");
    return `<div class="cards ${cols || "c3"}">${inner}</div>`;
  }

  function kpiRow(items) {
    const inner = (items || [])
      .map((k) =>
        `<div class="kpi"><div class="num${k.accent ? " accent" : ""}">${esc(k.num)}</div>` +
        `<div class="lbl">${esc(k.label)}</div></div>`
      )
      .join("");
    return `<div class="kpi-row">${inner}</div>`;
  }

  function table(t) {
    const head = `<tr>${(t.headers || []).map((h) => `<th>${esc(h)}</th>`).join("")}</tr>`;
    const body = (t.rows || [])
      .map((r) =>
        `<tr${r.highlight ? ' class="highlight"' : ""}>` +
        (r.cells || []).map((c) => `<td>${esc(c)}</td>`).join("") + `</tr>`
      )
      .join("");
    return `<table>${head}${body}</table>`;
  }

  function figBlock(s) {
    return `<div class="fig"><img src="${esc(figUrl(s.img))}" alt="">` +
      (s.caption ? `<div class="cap">${esc(s.caption)}</div>` : "") + `</div>`;
  }

  // --- Lamina completa ------------------------------------------------------

  function renderSlide(s) {
    const sec = s.sec ? ` data-sec="${esc(s.sec)}"` : "";
    let cls = "";
    let body = "";

    switch (s.layout) {
      case "cover": {
        cls = ' class="cover-slide"';
        const brand = s.brand ? `<h1 class="cover-brand">${esc(s.brand)}</h1>` : title(s, "h1");
        const sub = s.subtitle ? `<p class="cover-sub">${esc(s.subtitle)}</p>` : paras(s.paras);
        const meta = (s.meta || []).length
          ? `<div class="cover-meta">${s.meta.map((m) => `<div>${esc(m)}</div>`).join("")}</div>`
          : "";
        const img = s.img ? figUrl(s.img) : "assets/figs/cover.png";
        body =
          `<div class="cover"><div class="cover-text">${brand}${sub}${meta}</div>` +
          `<div class="cover-img" style="background-image:url('${img}')"></div></div>`;
        break;
      }

      case "divider":
        cls = ' class="divider"';
        body =
          `<div class="divider-grid"><div class="divider-text">` +
          (s.sec ? `<div class="sec-num">${esc(s.sec)}</div>` : "") +
          kicker(s) + title(s) + paras(s.paras) +
          `</div><div class="divider-art"><img src="${esc(figUrl(s.img))}" alt=""></div></div>`;
        break;

      case "closing":
        cls = ' class="closing center"';
        body = kicker(s) + title(s, "h1") + paras(s.paras);
        break;

      case "kpi":
        cls = ' class="center"';
        body = kicker(s) + title(s) + kpiRow(s.items) +
          (s.cards && s.cards.length ? cards(s.cards, "c2") : "");
        break;

      case "cards":
        body = kicker(s) + title(s) + cards(s.items, s.cols);
        break;

      case "table":
        body = kicker(s) + title(s) + table(s.table) + paras(s.paras);
        break;

      case "fig":
        body = kicker(s) + title(s) + figBlock(s) + bullets(s.bullets);
        break;

      case "twocol": {
        // Sin imagen real, se degrada a layout de texto (no <img src=""> roto).
        if (!s.img) {
          body = kicker(s) + title(s) + paras(s.paras) + bullets(s.bullets);
          break;
        }
        const wide = s.side === "right" ? " wide-right" : (s.side === "left" ? " wide-left" : " wide-right");
        const textCol = `<div class="col-text">${title(s)}${paras(s.paras)}${bullets(s.bullets)}</div>`;
        const artCol = `<div><img src="${esc(figUrl(s.img))}" alt="">` +
          (s.caption ? `<div class="cap">${esc(s.caption)}</div>` : "") + `</div>`;
        body = kicker(s) +
          `<div class="two-col${wide}">` +
          (s.side === "left" ? textCol + artCol : artCol + textCol) + `</div>`;
        break;
      }

      default: // text
        body = kicker(s) + title(s) + paras(s.paras) + bullets(s.bullets);
    }
    return `<section${cls}${sec}>${body}</section>`;
  }

  // --- Carga, render, switch ------------------------------------------------

  let revealReady = false;

  async function loadLang(lang) {
    const res = await fetch(`content/${lang}.json`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`no se pudo cargar content/${lang}.json`);
    return res.json();
  }

  function renderInto(slidesData) {
    const host = document.querySelector(".reveal .slides");
    host.innerHTML = slidesData.slides.map(renderSlide).join("\n");
  }

  function markActive(lang) {
    document.querySelectorAll("#lang-toggle button").forEach((b) =>
      b.classList.toggle("active", b.dataset.lang === lang)
    );
    document.documentElement.setAttribute("lang", lang);
    document.documentElement.setAttribute("data-lang", lang);
  }

  function getInitialLang() {
    const q = new URLSearchParams(location.search).get("lang");
    if (q === "es" || q === "en") return q;
    const saved = localStorage.getItem("agrosat_lang");
    if (saved === "es" || saved === "en") return saved;
    return (navigator.language || "es").toLowerCase().startsWith("en") ? "en" : "es";
  }

  async function setLang(lang) {
    // Preservar la lamina actual: re-renderizamos el mismo indice tras cambiar
    // el idioma (las laminas estan alineadas 1:1 entre es.json y en.json).
    const pos = revealReady && window.Reveal ? Reveal.getIndices() : { h: 0, v: 0 };
    const data = await loadLang(lang);
    renderInto(data);
    localStorage.setItem("agrosat_lang", lang);
    markActive(lang);
    if (revealReady && window.Reveal) {
      Reveal.sync();
      Reveal.slide(pos.h, pos.v);
    }
  }

  async function init() {
    const lang = getInitialLang();
    const data = await loadLang(lang);
    renderInto(data);
    markActive(lang);

    Reveal.initialize({
      hash: true, slideNumber: "c/t", transition: "slide", transitionSpeed: "default",
      width: 1600, height: 900, margin: 0, minScale: 0.2, maxScale: 2.0,
      center: false, controls: true, progress: true,
      plugins: window.RevealNotes ? [RevealNotes] : [],
    });
    revealReady = true;

    const toggle = document.getElementById("lang-toggle");
    if (toggle) {
      toggle.addEventListener("click", (e) => {
        const l = e.target && e.target.dataset && e.target.dataset.lang;
        if (l) setLang(l);
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
