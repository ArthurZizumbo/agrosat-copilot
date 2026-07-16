/* ============================================================================
   Motor de render de la presentacion AgroSatCopilot.
   El CONTENIDO vive en content/es.json y content/en.json (solo texto + figuras).
   Este motor construye los <section> de Reveal.js desde el idioma activo, y el
   switch de idioma re-renderiza desde el otro JSON. Asi, corregir un texto =
   editar una linea de JSON, sin tocar el HTML.
   ============================================================================ */
(function () {
  "use strict";

  // El motor de render vive en slides.js (compartido con print.html), para que
  // el PDF sea identico a la pantalla. Aqui solo se orquesta Reveal + i18n.
  const renderSlide = window.AgroDeck.renderSlide;

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
    // Preservar la lamina actual y cambiar el idioma SIN animacion (instantaneo):
    // las laminas estan alineadas 1:1 entre es.json y en.json, asi que solo
    // intercambiamos el texto de la misma lamina. Desactivamos la transicion
    // durante el swap para que no se vea un "deslizamiento" al traducir.
    const pos = revealReady && window.Reveal ? Reveal.getIndices() : { h: 0, v: 0 };
    const data = await loadLang(lang);
    if (revealReady && window.Reveal) {
      Reveal.configure({ transition: "none" });
    }
    renderInto(data);
    localStorage.setItem("agrosat_lang", lang);
    markActive(lang);
    if (revealReady && window.Reveal) {
      Reveal.sync();
      Reveal.slide(pos.h, pos.v);
      // restaurar la transicion normal para la navegacion entre laminas
      setTimeout(() => Reveal.configure({ transition: "fade" }), 50);
    }
  }

  async function init() {
    const lang = getInitialLang();
    const data = await loadLang(lang);
    renderInto(data);
    markActive(lang);

    Reveal.initialize({
      hash: true, slideNumber: "c/t", transition: "fade", transitionSpeed: "default",
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
